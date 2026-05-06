# NeuroQuant — The Seven Waves, in Plain English

This document explains, in non-technical language, what each of the seven hardening waves did, *why* it was necessary, and *what would have gone wrong if it had been skipped*.

The framework existed before Wave 1 — but it was a research prototype, not a deployable product. The waves turned it into something that can be installed, audited, and trusted.

---

## Why seven waves, instead of just "fix everything"?

The project had roughly 25 distinct production-readiness problems, ranging from security holes to incorrect math. Trying to fix all of them at once would have been chaotic — every change would touch every file, and it would be impossible to know which fix caused which test to break.

The seven-wave structure is a discipline:

1. Each wave focuses on **one theme** (security, real INT inference, reporting, etc.).
2. Each wave ends with **its own test suite** that proves the wave's claims.
3. No wave can break a previous wave's tests — the suite grows monotonically.

So if Wave 4's ONNX work breaks Wave 2's QAT path, you find out immediately. The waves are checkpoints in a long refactor.

---

## Wave 1 — Foundation: security, determinism, no cheating

### What it did

Three problems were addressed:

**Security.** The original code saved trained models with Python's `pickle` format. When you load a pickled model, Python executes whatever code is inside it. If anyone ever shared a checkpoint downloaded from the internet, an attacker could embed code that wipes the user's hard drive — and PyTorch would silently run it. Wave 1 banned the unsafe load path entirely; every checkpoint now loads in "tensors-only" mode where no code can run.

**Determinism.** Setting a random seed wasn't actually enough to make runs reproducible. PyTorch's GPU code uses non-deterministic algorithms by default for speed, so two runs with the same seed could produce slightly different results. Wave 1 forces every layer of the stack — Python's hash, NumPy, PyTorch, CUDA, cuDNN — into deterministic mode.

**No data leakage.** The framework was using the same dataset to (a) drive the search for good quantization configurations and (b) report the final accuracy. That's like grading your own homework — you get high marks but they don't predict how you'll do on the real exam. Wave 1 created four separate datasets with explicit roles: training, search, validation, and a held-out test set whose numbers are the only ones reported publicly.

### Why it mattered

Without security: the framework couldn't be shared, downloaded, or used by anyone who didn't write it themselves. Any GitHub user pulling a checkpoint would be installing a remote-code-execution backdoor.

Without determinism: nobody could verify the project's results. "I got 89.5% accuracy" is a meaningless claim if running the same code on the same machine gives a different number tomorrow.

Without split isolation: every accuracy number in the project was inflated. The framework was reporting numbers from a dataset it had been *optimised against*, not numbers from a clean test it had never seen.

### What would have happened without it

The framework would have been a research toy with impressive-looking but fundamentally untrustworthy numbers, that nobody could safely install or reproduce.

---

## Wave 2 — Real quantization-aware training (instead of pretend)

### What it did

The previous QAT (quantization-aware training) was a simulation. It "fake-quantized" weights inside the training loop but didn't connect properly to PyTorch's automatic differentiation system, and it kept activations as 32-bit floats. So the model that came out of training looked great on paper, but the *deployed* model — with real INT8 weights *and* real INT8 activations — behaved noticeably differently.

Wave 2 made QAT match what actually gets deployed:

- **Conv-BN folding.** Real INT8 deployment fuses each Conv2d layer with its following BatchNorm layer into a single operation. The previous QAT didn't do this, so the trained graph and the deployed graph were structurally different.

- **Real INT8 activations.** Activations are now actually 8-bit during training, not 32-bit floats pretending to be 8-bit. This is what the deployment hardware expects.

- **Proper gradient flow.** Quantization rounds numbers, which has a derivative of zero almost everywhere — so naively, no gradient flows through. Wave 2 implements the "straight-through estimator," which is the standard mathematical trick for letting gradients flow through round operations as if they weren't there.

- **Knowledge distillation.** During QAT, the model learns simultaneously from (a) the labels and (b) the predictions of the original 32-bit model. This recovers accuracy that would otherwise be lost.

- **AdaRound traversal order fix.** AdaRound is a fine-tuning technique applied layer by layer. It was being applied in parallel — each layer optimised assuming all other layers were perfect. Real deployment is sequential: errors accumulate. Wave 2 changed AdaRound to traverse layers in input-to-output order so each layer learns to compensate for the errors of the layers before it.

### Why it mattered

The mismatch between simulated QAT and deployed model could be 1–3% accuracy drop *just from the simulation gap*. So the published research result of "QAT loses only 0.5% accuracy" turned into "deployed model loses 2–3% accuracy" — a different conclusion entirely.

### What would have happened without it

Anyone who deployed the QAT models to real hardware would have gotten worse accuracy than the framework reported. The reports would have been technically true (the simulation did show those numbers) but practically misleading.

---

## Wave 3 — Audit the methods that were quietly broken

### What it did

Three of the framework's flagship quantization methods had problems.

- **AWQ was mathematically wrong.** AWQ scales weights by an activation-dependent factor, with a corresponding scaling on the input. The previous implementation applied the weight scaling but *forgot the input compensation*. The forward pass produced numerical garbage. Wave 3 rewrote AWQ from scratch with the correct math, verified by a forward-equivalence test.

- **SmoothQuant used a single global parameter.** SmoothQuant has a tunable parameter α that controls how much "difficulty" to migrate from activations to weights. The original code used the same α for every layer in the network. But different layers want different α values. Wave 3 added a per-layer search that picks each layer's α individually.

- **No combined method.** Two of the methods (SmoothQuant and GPTQ) were known to work better together than separately — this combination is the standard production recipe in 2024+. The framework had each method independently but no way to chain them. Wave 3 added the combined method.

- **Fisher estimator.** The framework computed per-layer sensitivity using a slow second-order method. Wave 3 added a faster Fisher-information-based estimator that's about 3× quicker and gives almost identical results.

### Why it mattered

Without the AWQ fix: every result the project published using AWQ was wrong. Not approximately wrong — *fundamentally* wrong, because the math didn't work.

Without per-layer SmoothQuant: results were 0.5–1% worse than they could have been on every model.

Without the combined method: the framework couldn't reproduce the state-of-the-art numbers from recent papers, because those papers use the combined approach.

### What would have happened without it

The framework would have shipped with a quantization method (AWQ) whose published numbers were unreliable, and would have lost to any competing framework using the standard combined recipe.

---

## Wave 4 — Stop simulating; measure the real thing

### What it did

Until Wave 4, "INT8" was a research fiction. The framework simulated INT8 quantization by storing 32-bit floats that took only 256 distinct values. The "model size" it reported was a calculation: number of parameters × bitwidth ÷ 8. The "latency" was measured running this 32-bit simulation through PyTorch.

None of those numbers had any direct relationship to what would happen at deployment.

Wave 4 wired the framework to ONNX, the actual deployment runtime:

- **Real INT8 export.** Every quantized model is now written to disk as a true INT8 ONNX file — the same kind of file production servers actually load.

- **Real disk size.** The "model size" reported is the literal size of the .onnx file in megabytes. Not a calculation — a measurement.

- **Real latency.** The "latency" reported is measured by running the model through ONNX Runtime, the production inference engine.

- **Hardware-aware search.** A new optional mode profiles the per-layer latency cost on the deployment runtime ahead of time, then lets the search optimise for accuracy *and* size *and* latency simultaneously, instead of just the first two.

### Why it mattered

The synthetic numbers and the real numbers can disagree by 30% or more. A model the framework reports as "2.3 MB" might actually be 1.7 MB on disk because of compiler-level packing. A latency reported as "5 ms" might actually be 8 ms under ONNX Runtime because of overhead the simulation didn't account for.

These aren't small differences — they change which model wins the comparison. Wave 4 also exposed a counterintuitive truth: on small layers, INT8 inference can be *slower* than FP32 because of the per-call setup overhead. The hardware-aware search now sees this and avoids it. The simulated search couldn't have known.

### What would have happened without it

The framework's claim of being a deployment-ready quantization tool would have been a marketing lie. Users would optimise for the wrong objective and discover the truth only after deploying to production.

---

## Wave 5 — Make the reports honest

### What it did

After Wave 4, the framework was *measuring* real numbers but still *displaying* synthetic ones. The public report and MLflow logs hadn't been updated.

Wave 5 fixed every output channel:

- **Headline table.** The end-of-pipeline summary table now shows three new columns: real ONNX file size, real ORT latency, real ORT throughput. The synthetic numbers are still there for ablation but clearly labelled as "theoretical."

- **Pareto plots.** The accuracy-vs-size scatter plot uses the real on-disk size. A new 3-D plot adds latency as a third axis when hardware-aware mode is on.

- **Deployment-fidelity section.** The report now ends with a section that compares the synthetic estimates to the real measurements: "the on-disk size is on average X% smaller than the theoretical estimate," "the median quantized model runs 3.2× faster than the FP32 baseline."

- **Reproducibility manifest.** Each run writes a JSON file recording everything needed to reproduce the result: Python version, all package versions including ONNX Runtime, the exact hardware, the random seed, the size and latency of the FP32 baseline.

- **MLflow integration.** Every quantized model's .onnx file is uploaded to MLflow as a downloadable artefact, with all the real numbers logged as metrics.

### Why it mattered

Wave 4 made the framework honest internally. Wave 5 made it honest externally. Without it, the framework would have been doing the right thing internally but reporting the wrong thing to its users — the worst kind of bug because it's invisible.

### What would have happened without it

Users would have continued to make decisions based on synthetic numbers, even though the framework was producing real ones. The work in Wave 4 would have been wasted.

---

## Wave 6 — Make the project testable, validatable, and CI-ready

### What it did

The framework had tests, but they were ad-hoc and inconsistent. Different test files defined the same model class differently. There was no coverage measurement, no continuous integration, and no integration test that proved the whole pipeline worked end-to-end.

The configuration system was also fragile: bad values in `config.yaml` (a typo, a negative number, a string where an integer was expected) would fail deep inside a phase with a confusing traceback that didn't point at the offending field.

Wave 6 hardened the development loop:

- **Shared test fixtures.** A single `conftest.py` defines the standard test model, calibration data, and config — every test file uses these instead of reinventing them.

- **Coverage gate.** Every commit is now measured. The continuous-integration workflow rejects pull requests that drop coverage below 80%. (The project sits at 81.3%.)

- **Integration smoke.** A new test runs the *entire* 9-phase pipeline on a tiny synthetic dataset in about 60 seconds. If any phase doesn't hand off correctly to the next phase, this test catches it.

- **Property-based tests.** Instead of testing individual examples ("when input is 5, output should be 10"), these tests state mathematical *properties* ("for any input, the output should be smaller than the input") and let a library generate random inputs to try to break the property. They catch edge cases the developer didn't think of.

- **Pydantic configuration.** Configuration values are now validated at the moment they're loaded, not deep inside a phase. A typo like `device: "tpu"` instead of `device: "cuda"` fails immediately with a clear error message.

- **Continuous integration.** A GitHub Actions workflow runs the full test suite on Linux against Python 3.10, 3.11, and 3.12 for every push and pull request. A regression that passes locally but breaks on CI is caught before merge.

### Why it mattered

Without these, future changes to the project — by anyone, including the original author — would slowly degrade quality. Coverage would drop without anyone noticing. New phases would integrate poorly with old ones. Bad config values would produce mysterious crashes. The framework would be locked in to its first-version author.

### What would have happened without it

The project would have been a snapshot of one person's work that nobody else could safely modify. Any second contributor would have been stuck guessing whether their changes broke something.

---

## Wave 7 — Packaging and documentation

### What it did

The framework worked, but using it required cloning the source code, knowing about `main.py`, and reading scattered notes to figure out the configuration format.

Wave 7 packaged it:

- **Pip installable.** `pip install neuroquant` (once published) now installs the framework like any standard Python package.

- **Console script.** Users type `neuroquant --config config.yaml` instead of `python /path/to/main.py --config config.yaml`. The command is on their PATH.

- **README.** The project root now has a proper README with badges, install instructions, run examples, a methods table, and links to detailed architecture docs.

- **Per-wave architecture docs.** The seven-wave decision history is preserved as seven markdown files under `docs/architecture/`, each with the original decision matrix, what shipped, and what tests guard it.

- **License file.** MIT license, matching the declaration in the package metadata.

### Why it mattered

A framework nobody can install isn't a framework — it's a personal script with delusions of grandeur. Wave 7 turned the codebase into something a stranger can clone, install, run, and verify in five minutes.

### What would have happened without it

The project would have been undocumented research code. A graduation defense committee asking "how would someone else use this?" would have gotten "you'd have to ask me" as the only honest answer.

---

## The cumulative effect

Each wave on its own is an incremental improvement. Together they cross a threshold:

- **Before Wave 1**, the framework's results were unreliable.
- **After Wave 2**, the trained models matched what would actually deploy.
- **After Wave 3**, the methods were mathematically correct.
- **After Wave 4**, the reported numbers were measured on real deployment infrastructure.
- **After Wave 5**, those measurements actually appeared in the reports.
- **After Wave 6**, the project couldn't silently regress.
- **After Wave 7**, anyone could install and use it.

You could call this "production-grade." You could equally call it "honest." The two turn out to mean the same thing: the framework does what it claims, can be verified by anyone, and won't quietly fail when used in a setting different from the one it was developed in.

That's the difference between a research prototype and a deployable system.
