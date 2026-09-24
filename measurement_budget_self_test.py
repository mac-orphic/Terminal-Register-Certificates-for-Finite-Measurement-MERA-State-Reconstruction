#!/usr/bin/env python3
"""Fast implementation checks for the measurement-budget extension."""

from __future__ import annotations

import json

import numpy as np

from qipmera.budget import checkpointed_spsa_train, terminal_certificate_samples
from qipmera.core import architecture, random_parameters, spsa_train


def main() -> None:
    n = 4
    target = np.zeros(1 << n, complex)
    target[0] = 1.0
    spec = architecture(n, "MPS")
    initial = random_parameters(spec, 123)
    stages = [
        {"steps": 2, "directions": 1, "shots": 10, "learning_rate": 0.01, "perturbation": 0.05}
    ]
    measurement_seed = 456
    direction_seed = 789

    reference_theta, _, reference_resources = spsa_train(
        target,
        n,
        spec,
        initial,
        stages,
        measurement_seed,
        direction_seed,
    )
    checkpoints, checkpoint_resources = checkpointed_spsa_train(
        target,
        -3.0,
        n,
        1.0,
        spec,
        initial,
        stages,
        [20, 40],
        measurement_seed,
        direction_seed,
    )
    checkpoint_theta = np.asarray(checkpoints[-1]["trained_parameters"])

    certificate = terminal_certificate_samples(
        p0=0.9,
        infidelity=0.1,
        certificate_shots=1000,
        delta=0.05,
        tau=0.0,
        epsilon_top=0.0,
        repetitions=20,
        seed=321,
    )
    mera = architecture(16, "MERA_PM225")
    mps = architecture(16, "MPS")
    checks = {
        "checkpoint_budgets_exact_for_toy_case": [row["achieved_training_shots"] for row in checkpoints] == [20, 40],
        "checkpoint_trajectory_matches_original_spsa": bool(np.allclose(reference_theta, checkpoint_theta, atol=1e-12, rtol=0)),
        "checkpoint_resources_match_original": checkpoint_resources["training_shots"] == reference_resources["training_shots"] == 40,
        "certificate_sample_count": len(certificate) == 20,
        "certificate_bounds_in_unit_interval": all(0.0 <= row["certificate_upper_bound"] <= 1.0 for row in certificate),
        "equal_parameter_count": mera.coordinates == mps.coordinates == 225,
        "gate_exposure_is_not_matched": mera.physical_blocks == 26 and mps.physical_blocks == 15,
    }
    report = {"all_checks_passed": all(checks.values()), "checks": checks}
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
