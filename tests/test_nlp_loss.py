"""Smoke tests for the NLP loss bridge.

Does NOT pull HuggingFace transformers — instead uses a stub model
that mimics the HF API (``forward(**inputs, labels=...) -> object
with .loss``). This isolates the loss-bridge plumbing from the heavy
HF install, so the test runs on every CI machine, not just the ones
with the [nlp] extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn

from neuroquant import QuantizationConfig


@dataclass
class _StubHFOutput:
    """Mimic ``transformers.modeling_outputs.SequenceClassifierOutput``."""
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class _StubHFModel(nn.Module):
    """Behaviourally compatible with HF text-classification models.

    Accepts ``input_ids`` + ``attention_mask`` via ``**kwargs`` and
    returns a stub output carrying ``.loss`` and ``.logits``.
    """

    def __init__(self, vocab_size: int = 100, num_labels: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)
        self.cls = nn.Linear(8, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> _StubHFOutput:
        pooled = self.embed(input_ids).mean(dim=1)
        logits = self.cls(pooled)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return _StubHFOutput(loss=loss, logits=logits)


def test_nlp_task_accepted_by_config() -> None:
    cfg = QuantizationConfig(task="nlp")
    assert cfg.task == "nlp"


def test_nlp_loss_bridge_returns_loss_from_dict_input() -> None:
    """Exercises ``_build_task_loss_fn`` for ``task='nlp'`` directly.

    We bypass the full CLI pipeline (which would need a real dataset
    + tokenizer) by constructing the bridge through a minimal class
    that mimics ``NeuroQuantPipeline._build_task_loss_fn``'s contract.
    """
    # Reuse the actual implementation from the pipeline class so the
    # test pins the real bridge, not a copy.
    from neuroquant.cli import NeuroQuantPipeline

    cfg = QuantizationConfig(task="nlp")

    # We don't want to instantiate the whole pipeline (it touches
    # tracking + checkpointing); instead snapshot the loss-fn builder
    # by binding it to a minimal object that just carries the config.
    class _MiniPipe:
        config = cfg
    loss_fn = NeuroQuantPipeline._build_task_loss_fn(_MiniPipe())  # type: ignore[arg-type]

    model = _StubHFModel()
    x = {
        "input_ids": torch.randint(0, 100, (4, 16)),
        "attention_mask": torch.ones(4, 16, dtype=torch.long),
    }
    y = torch.randint(0, 2, (4,))

    loss = loss_fn(model, x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)


def test_nlp_loss_bridge_rejects_non_dict_input() -> None:
    """A tensor input under ``task='nlp'`` must raise a clear error."""
    from neuroquant.cli import NeuroQuantPipeline

    cfg = QuantizationConfig(task="nlp")

    class _MiniPipe:
        config = cfg
    loss_fn = NeuroQuantPipeline._build_task_loss_fn(_MiniPipe())  # type: ignore[arg-type]

    model = _StubHFModel()
    with pytest.raises(TypeError, match="dict inputs"):
        loss_fn(model, torch.zeros(4, 16), torch.zeros(4, dtype=torch.long))
