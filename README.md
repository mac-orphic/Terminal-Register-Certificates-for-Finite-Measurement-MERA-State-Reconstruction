# Terminal-Register Certification for Finite-Measurement Learning of Multiscale Quantum States

This repository contains the reproducible simulation code for learning one-dimensional transverse-field Ising model (TFIM) ground states with binary MERA and for comparing MERA with matrix-product-state (MPS), tree-tensor-network (TTN), and local-circuit baselines.

The central contribution is a **terminal-register certificate**: after a fixed analysis circuit, all discarded registers are measured jointly at the end of the circuit. Their all-zero probability gives a finite-measurement certificate for reconstruction fidelity without sequential postselection.

The final fairness extension focuses on three questions:

1. Does MERA's long-range advantage survive **matched aggregate gate exposure**?
2. Does it persist under **exact-objective multistart optimization**?
3. Does it survive comparison with a **causal-cone-complete local circuit** having the same number of independent coordinates?

## Main result in one paragraph

At `n=16` and fixed fields `h/J ∈ {0.9, 1.0, 1.1}`, the gate-exposure-matched comparison does **not** resolve a fidelity or energy advantage between the 225-coordinate MERA and MPS. It does, however, favor MERA for held-out long-range `ZZ` error and half-chain entropy error at the largest exposure. The exact-objective multistart diagnostic favors MERA in 58/60 paired fidelity comparisons and 60/60 long-range comparisons, but none of the 120 runs satisfies the preregistered stationarity test; these are therefore **best-found optimization results, not certified global optima**. Against a nine-layer, causal-cone-complete, 225-coordinate local circuit with greater aggregate gate exposure, MERA wins all 30 paired comparisons on fidelity, long-range error, energy error, and entropy error.

These results support a **resource-qualified multiscale-geometry advantage**. They do not establish universal MERA superiority, a hardware-runtime advantage, or global optimality.

## Repository contents

```text
.
├── qipmera/
│   ├── budget.py                  # Resource and shot accounting
│   ├── core.py                    # TFIM states, circuit models, observables
│   └── exact_torch.py             # Exact-objective PyTorch backend
├── protocols/
│   ├── gate_exposure_v1.json      # Frozen gate-exposure protocol
│   ├── noiseless_multistart_v1.json
│   ├── causal_cone_local_v1.json
│   ├── measurement_budget_v1.json # Supporting shot-budget experiment
│   └── resource_frontier_v2.json  # Earlier resource-frontier study
├── run_gate_exposure.py
├── run_noiseless_multistart.py
├── run_causal_cone_local.py
├── run_measurement_budget.py
├── run_frontier.py
├── analyze_measurement_budget.py
├── analyze_frontier.py
├── structural_self_test.py
├── torch_backend_self_test.py
├── measurement_budget_self_test.py
├── analysis_self_test.py
├── self_test.py
├── smoke_test.py
├── requirements.txt
└── results/
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <YOUR-REPOSITORY-URL>
cd mera_fairness_extension_v1

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies are NumPy, SciPy, pandas, Matplotlib, and PyTorch. A CUDA-capable PyTorch installation is optional and is used only by the exact-objective multistart study.

## Validate the implementation first

Run the structural and exact-backend tests before launching any experiment:

```bash
python structural_self_test.py
python torch_backend_self_test.py
```

Then validate all three locked protocols without performing optimization:

```bash
python run_gate_exposure.py --dry-run
python run_noiseless_multistart.py --dry-run
python run_causal_cone_local.py --dry-run
```

All checks should report `all_checks_passed: true`.

## Reproduce the primary fairness studies

The frozen execution order is:

1. aggregate gate-exposure comparison;
2. exact-objective multistart diagnostic;
3. causal-cone-complete local control.

Do not modify a frozen protocol after inspecting its results. Failed or nonstationary runs must be retained.

### 1. Equal aggregate gate exposure: MERA-225 versus MPS-225

```bash
python run_gate_exposure.py --workers 2
```

Use `--workers 1` on a memory-limited machine.

The experiment uses:

- `n = 16`;
- `h/J ∈ {0.9, 1.0, 1.1}`;
- paired evaluation replicates `80–89`;
- 225 independent coordinates for both architectures;
- aggregate SU(4)-shot-gate exposures
  `2.6×10^7`, `2.6×10^8`, `2.6×10^9`, and `8.4×10^9`.

There are 60 optimizer trajectories—two architectures, three fields, and ten paired replicates—producing 240 checkpoint-level result rows across four exposure endpoints.

Primary outputs are written to `results/gate_exposure_v1/`:

- `analysis.json`
- `summary.csv`
- `paired_contrasts.csv`
- `execution_manifest.json`
- `fidelity_vs_gate_exposure.pdf`
- `long_range_error_vs_gate_exposure.pdf`

### 2. Exact-objective multistart diagnostic

Automatic device selection:

```bash
python run_noiseless_multistart.py --device auto --workers 1
```

CPU-only alternative:

```bash
python run_noiseless_multistart.py --device cpu --workers 2
```

Kaggle or another CUDA runtime:

```bash
python run_noiseless_multistart.py --device cuda --workers 1
```

The diagnostic runs 20 paired restarts per field and architecture, for 120 runs in total. It uses the exact `eta_sum` objective, Adam, 5,000 steps, and the frozen two-stage learning-rate schedule. Its role is to probe finite-shot versus optimization limitations—not to prove a global optimum.

Outputs are written to `results/noiseless_multistart_v1/`.

### 3. Causal-cone-complete local control

```bash
python run_causal_cone_local.py --workers 2
```

This compares:

- `MERA_PM225`: 26 physical two-qubit blocks and 225 independent coordinates;
- `LOCAL_LC225`: nine layers, 68 physical two-qubit blocks, and 225 independent coordinates.

Nine local layers are the minimum required by this construction to cover all 36 held-out long-range pairs; the corresponding eight-layer control misses the pair `(0,15)`. The local model therefore removes the disconnected-light-cone objection while using about `2.62×` the aggregate gate exposure of MERA.

Outputs are written to `results/causal_cone_local_v1/`.

## Final verified results

All differences below are `MERA − baseline`. Positive fidelity differences favor MERA; negative error differences favor MERA. The two primary architecture contrasts use 97.5% paired bootstrap intervals so that the two-comparison family has 95% coverage under Bonferroni correction.

### Gate-exposure-matched endpoint

Largest aggregate exposure: `G = 8.4×10^9`.

| Metric | MERA-225 | MPS-225 | Paired difference | 97.5% interval | Interpretation |
|---|---:|---:|---:|---:|---|
| Fidelity | 0.953529 | 0.952925 | +0.000604 | [-0.016347, 0.015413] | Unresolved |
| Held-out long-range `ZZ` MAE | 0.069314 | 0.083896 | -0.014582 | [-0.028112, -0.000340] | Favors MERA |
| Half-chain entropy error | 0.028495 | 0.058940 | -0.030444 | [-0.048778, -0.010442] | Favors MERA |
| Energy-density error | 0.009409 | 0.008788 | +0.000621 | [-0.001672, 0.003054] | Unresolved |

The estimand is paired optimizer-seed performance averaged over the three fixed TFIM fields. It is not an estimate of performance over arbitrary fields.

### Exact-objective multistart diagnostic

| Outcome | Result |
|---|---:|
| Paired fidelity wins for MERA | 58/60 |
| Paired long-range-error wins for MERA | 60/60 |
| Runs satisfying the full stationarity criterion | 0/120 |

The diagnostic shows a consistent best-found separation, but the failed stationarity audit prevents any global-optimum or best-achievable-capacity claim.

### Causal-cone-complete local control

Thirty paired comparisons across the three fixed fields:

| Metric | Pooled paired difference | 97.5% interval | MERA wins |
|---|---:|---:|---:|
| Fidelity | +0.251757 | [0.243706, 0.260153] | 30/30 |
| Held-out long-range `ZZ` MAE | -0.105822 | [-0.116590, -0.095612] | 30/30 |
| Energy-density error | -0.043510 | [-0.045970, -0.041217] | 30/30 |
| Half-chain entropy error | -0.252575 | [-0.280188, -0.225882] | 30/30 |

## Certificate implemented by the project

For a fixed analysis circuit, let `p0` be the probability that every discarded register is measured as zero, and let `F_top` be the fidelity of the retained top state conditioned on that outcome. The reconstruction identity is

$$
F = p_0 F_{\mathrm{top}}.
$$

Consequently,

$$
1-F = (1-p_0)+p_0(1-F_{\mathrm{top}})
\le (1-p_0)+\epsilon_{\mathrm{top}}.
$$

With `N` independent terminal measurements, empirical all-zero frequency `\widehat p_0`, confidence failure probability `\delta`, and a calibrated total-variation implementation budget `\tau`, the finite-measurement bound used in the manuscript is

$$
1-F
\le
1-\widehat p_0
+\sqrt{\frac{\log(1/\delta)}{2N}}
+\tau
+\epsilon_{\mathrm{top}}
$$

with probability at least `1-\delta`, under the stated fixed-circuit and implementation-error assumptions. The full assumptions and proofs—including ideal sequential/terminal equivalence and inverse-acceptance postselection instability—are given in the accompanying manuscript.

## Supporting experiments

The repository also contains the earlier measurement-budget and resource-frontier studies. They are useful for diagnostics and historical comparison, but the three fairness experiments above provide the primary evidence for the final resource-qualified architecture claims.

Validate the supporting pipelines with:

```bash
python measurement_budget_self_test.py
python analysis_self_test.py
python self_test.py
python smoke_test.py
```

Inspect their command-line options before running:

```bash
python run_measurement_budget.py --help
python run_frontier.py --help
```

## Running on Kaggle

Upload the repository as a Kaggle dataset or notebook archive, enable a GPU only for the exact-objective study, and run:

```python
%cd /kaggle/working/mera_fairness_extension_v1
!pip install -q -r requirements.txt

!python structural_self_test.py
!python torch_backend_self_test.py

!python run_gate_exposure.py --workers 2
!python run_noiseless_multistart.py --device cuda --workers 1
!python run_causal_cone_local.py --workers 2
```

If Kaggle reports an out-of-memory error, reduce the non-CUDA commands to `--workers 1`. Download the entire `results/` directory after completion so that manifests, raw runs, summaries, and figures remain together.

## Reproducibility and interpretation rules

- Protocol JSON files marked frozen are part of the experimental record.
- Every evaluation replicate is reported; failed runs are not removed or rerun with changed hyperparameters.
- Seed namespaces are paired within each architecture comparison.
- The exact-objective study reports best-found multistart solutions because its stationarity criterion was not met.
- Aggregate SU(4)-shot-gate exposure is an architecture-neutral simulation accounting unit, not a complete estimate of compiled hardware runtime.
- Parameter matching, gate-exposure matching, causal-cone matching, and depth matching answer different fairness questions; the repository does not treat them as interchangeable.
- Experiments are classical simulations. No experimental-hardware performance claim is made.

## Expected runtime

Runtime depends strongly on CPU/GPU model, memory bandwidth, and worker count. The exact-objective multistart study is the most computationally demanding component. Use the dry-run commands to verify protocols before committing compute, and keep all execution manifests with the reported results.

## Citation

If this code is useful in your work, please cite the accompanying manuscript:

> *Terminal-Register Certification for Finite-Measurement Learning of Multiscale Quantum States.*

Author, venue, and archival identifiers should be added here when the public preprint is released.

## Contact and issue reports

For a reproducibility issue, open a GitHub issue and include:

- the command that was run;
- Python, PyTorch, and operating-system versions;
- CPU/GPU model;
- the relevant `execution_manifest.json`;
- the full traceback or failed self-test report.

