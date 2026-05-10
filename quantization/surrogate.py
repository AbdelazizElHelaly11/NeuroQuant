"""
NeuroQuant v2.0 — Surrogate-assisted NSGA-II (Phase 1c, BRP-NAS style).

The surrogate predicts ``accuracy_loss`` from a per-layer bitwidth
vector. Once trained, scoring 10 000 candidates costs ~10 ms, so each
NSGA generation can propose hundreds of children, surrogate-rank them,
and only spend real-evaluation budget on the top-K predicted picks.

Public contract:

* ``AccuracySurrogate(n_layers, hessian_features=None)``
    Build an empty surrogate. ``hessian_features`` is optional —
    when supplied, the per-layer Hessian sensitivity is folded into
    the input vector so the model can learn "INT4 on a sensitive
    layer is worse than INT4 on a robust one" without seeing that
    explicitly during search.

* ``surrogate.add_observation(bitwidth_vec, accuracy_loss)``
    Append one (X, y) pair to the training set.

* ``surrogate.fit()``
    Train (or retrain) on everything observed so far. No-op if
    fewer observations than ``min_train_samples`` exist or sklearn
    is unavailable.

* ``surrogate.predict(bitwidth_vec)``
    Return the predicted accuracy_loss for a single candidate.

* ``surrogate.score_batch(bitwidth_vecs)``
    Vectorised version. Returns a numpy array of predicted losses.

* ``surrogate.is_ready()``
    Whether the surrogate has been trained at least once and is
    safe to call ``predict`` / ``score_batch`` on.

The implementation deliberately avoids hard dependencies beyond what
the rest of NeuroQuant already pins (numpy + sklearn). XGBoost is
faster but adds a 200 MB transitive footprint; the project goal is a
single-pip-install workflow, so sklearn's ``GradientBoostingRegressor``
is the right ceiling. For the volumes used here (~hundreds of training
points, hundreds of predictions per generation) that's plenty.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("neuroquant")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optional dependency probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Surrogate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AccuracySurrogate:
    """Surrogate model that predicts accuracy_loss from a bitwidth vector.

    The model is an ensemble of decision trees (sklearn's
    GradientBoostingRegressor). It works well on the kind of
    discrete + monotone-ish features mixed-precision search produces
    and trains in milliseconds on the data volumes NSGA generates.

    Generic by construction:
      * No hardcoded layer count — initialised with ``n_layers``.
      * Optional Hessian features are accepted as a flat array;
        ``hessian_features.shape == (n_layers,)`` is the only
        assumption.
      * Falls back to a no-op stub when sklearn is unavailable so
        callers never crash on a missing import.
    """

    def __init__(
        self,
        n_layers: int,
        hessian_features: Optional[Sequence[float]] = None,
        *,
        min_train_samples: int = 20,
        retrain_every: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_layers = int(n_layers)
        self.min_train_samples = int(min_train_samples)
        self.retrain_every = int(retrain_every)
        self.random_state = int(random_state)

        # Normalise hessian features to log-space + z-score so the
        # surrogate sees them on the same scale as the (0, 1) bitwidth
        # bits. Magnitude differences of 5+ orders of magnitude in raw
        # Hessian values would otherwise dominate the splits.
        self._hessian_features: Optional[np.ndarray] = None
        if hessian_features is not None:
            arr = np.asarray(list(hessian_features), dtype=float)
            if arr.shape == (self.n_layers,):
                arr = np.log1p(np.maximum(arr, 0.0))
                std = float(arr.std()) or 1.0
                self._hessian_features = (arr - arr.mean()) / std
            else:
                logger.warning(
                    "AccuracySurrogate: hessian_features shape %s != "
                    "(%d,); ignoring.", arr.shape, self.n_layers,
                )

        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._model: Any = None
        self._evals_since_fit: int = 0
        self._available: bool = _has_sklearn()
        if not self._available:
            logger.warning(
                "scikit-learn not importable — surrogate disabled. "
                "Install with `pip install scikit-learn` to enable "
                "BRP-NAS style search acceleration."
            )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _featurise(self, bitwidth_vec: Sequence[int]) -> np.ndarray:
        """Convert one bitwidth vector to the surrogate's feature vector.

        Encoding rules (kept minimal so the surrogate generalises across
        models):
          * raw bit choice as 0/1 per layer (0 = INT4, 1 = INT8)
          * per-layer Hessian (z-scored log) when supplied
          * aggregate features: fraction of INT4 layers, fraction of
            high-sensitivity INT4 layers — give the booster easy
            signals about overall compression / risk.
        """
        bw = np.asarray(list(bitwidth_vec), dtype=float)
        if bw.shape != (self.n_layers,):
            raise ValueError(
                f"bitwidth vector shape {bw.shape} != ({self.n_layers},)"
            )
        # Ensure bits live in [0, 1]: convert {4, 8} → {0, 1} when
        # callers hand us literal bitwidths instead of bits.
        if bw.max() > 1.5:
            bw = (bw >= 8).astype(float)

        chunks: List[np.ndarray] = [bw]
        if self._hessian_features is not None:
            chunks.append(self._hessian_features)
            # Cross feature: bitwidth × sensitivity. INT4 on a HIGH
            # sensitivity layer is the canonical accuracy-killer.
            chunks.append((1.0 - bw) * self._hessian_features)

        # Aggregate features
        int4_frac = float(1.0 - bw.mean())
        if self._hessian_features is not None:
            risky_share = float(((1.0 - bw) * (self._hessian_features > 0)).mean())
        else:
            risky_share = int4_frac
        chunks.append(np.asarray([int4_frac, risky_share]))

        return np.concatenate(chunks)

    def _featurise_batch(
        self, bitwidth_vecs: Sequence[Sequence[int]],
    ) -> np.ndarray:
        return np.stack([self._featurise(v) for v in bitwidth_vecs])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observation(
        self, bitwidth_vec: Sequence[int], accuracy_loss: float,
    ) -> None:
        """Record a real evaluation result.

        ``accuracy_loss`` follows the same convention as NSGA: lower
        is better, may be negative (quantized model beats FP32 — KD
        distillation, regularisation by quantization noise, etc.).
        """
        if not self._available:
            return
        try:
            x = self._featurise(bitwidth_vec)
        except ValueError as exc:
            logger.debug("Surrogate skipped sample: %s", exc)
            return
        self._X.append(x)
        self._y.append(float(accuracy_loss))
        self._evals_since_fit += 1

    def needs_refit(self) -> bool:
        """Whether ``fit()`` should be called now."""
        return (
            self._available
            and len(self._y) >= self.min_train_samples
            and (self._model is None or self._evals_since_fit >= self.retrain_every)
        )

    def fit(self) -> bool:
        """Train (or retrain) the surrogate.

        Returns True when a model was fit, False otherwise (sklearn
        missing, too few samples, or training raised). Failures are
        logged at debug level — the caller falls back to plain NSGA.
        """
        if not self._available or len(self._y) < self.min_train_samples:
            return False
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            X = np.stack(self._X)
            y = np.asarray(self._y, dtype=float)
            n_estimators = int(min(200, max(50, len(y) * 2)))
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self._model = model
            self._evals_since_fit = 0
            logger.info(
                "  Surrogate fit on %d samples (n_estimators=%d).",
                len(y), n_estimators,
            )
            return True
        except Exception as exc:
            logger.debug("Surrogate fit failed: %s", exc)
            return False

    def is_ready(self) -> bool:
        return self._available and self._model is not None

    def predict(self, bitwidth_vec: Sequence[int]) -> Optional[float]:
        if not self.is_ready():
            return None
        try:
            x = self._featurise(bitwidth_vec).reshape(1, -1)
            return float(self._model.predict(x)[0])
        except Exception as exc:
            logger.debug("Surrogate predict failed: %s", exc)
            return None

    def score_batch(
        self, bitwidth_vecs: Sequence[Sequence[int]],
    ) -> Optional[np.ndarray]:
        if not self.is_ready() or not bitwidth_vecs:
            return None
        try:
            X = self._featurise_batch(bitwidth_vecs)
            return np.asarray(self._model.predict(X), dtype=float)
        except Exception as exc:
            logger.debug("Surrogate batch predict failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_observations(self) -> int:
        return len(self._y)

    def metrics(self) -> Dict[str, Any]:
        """Cheap leave-out diagnostics — never use on hot paths."""
        if not self.is_ready() or len(self._y) < 4:
            return {}
        try:
            from sklearn.metrics import mean_absolute_error
            X = np.stack(self._X)
            y = np.asarray(self._y, dtype=float)
            yhat = self._model.predict(X)
            return {
                "n_samples": len(y),
                "train_mae": float(mean_absolute_error(y, yhat)),
                "loss_min": float(y.min()),
                "loss_max": float(y.max()),
            }
        except Exception:
            return {}
