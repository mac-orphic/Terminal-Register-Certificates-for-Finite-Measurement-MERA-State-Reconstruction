#!/usr/bin/env python3
"""Synthetic end-to-end check of statistics and figure generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from analyze_measurement_budget import analyze


def main() -> None:
    root = Path(__file__).resolve().parent
    budgets = [1_000_000, 10_000_000, 100_000_000, 560_000_000]
    fields = [0.9, 1.0, 1.1]
    rows = []
    certificate = []
    for field in fields:
        for replicate in [0, 1]:
            for architecture in ["MERA_PM225", "MPS"]:
                for index, budget in enumerate(budgets):
                    advantage = 0.01 if architecture == "MERA_PM225" else 0.0
                    fidelity = 0.80 + 0.04 * index + advantage
                    rows.append({
                        "field": field,
                        "replicate": replicate,
                        "architecture": architecture,
                        "requested_training_shots": budget,
                        "achieved_training_shots": budget,
                        "fidelity": fidelity,
                        "zz_long_range_mae": 0.20 - 0.03 * index - advantage,
                        "energy_density_error": 0.10 - 0.01 * index,
                        "half_chain_entropy_error": 0.15 - 0.02 * index,
                        "joint_failure": 1.0 - fidelity,
                        "success_F_gt_0.99": int(fidelity > 0.99),
                        "physical_su4_blocks": 26 if architecture == "MERA_PM225" else 15,
                        "independent_coordinates": 225,
                        "abstract_su4_shot_gate_executions": budget * (26 if architecture == "MERA_PM225" else 15),
                        "modeled_entangling_shot_gate_executions": 3 * budget * (26 if architecture == "MERA_PM225" else 15),
                    })
    for architecture in ["MERA_PM225", "MPS"]:
        for budget in budgets:
            for shots in [100, 1000, 10000, 100000, 1000000]:
                certificate.append({
                    "architecture": architecture,
                    "requested_training_shots": budget,
                    "certificate_shots": shots,
                    "hoeffding_radius": (1.497866 / shots) ** 0.5,
                    "certificate_upper_bound": 0.2,
                    "covers_exact_infidelity": True,
                })

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        pd.DataFrame(rows).to_csv(output / "checkpoint_runs.csv", index=False)
        pd.DataFrame(certificate).to_csv(output / "certificate_samples.csv", index=False)
        result = analyze(output, root / "protocols" / "measurement_budget_v1.json")
        required = [
            "analysis.json",
            "budget_summary.csv",
            "paired_contrasts.csv",
            "certificate_summary.csv",
            "resource_audit.csv",
            "fidelity_vs_training_shots.pdf",
            "success_probability_vs_training_shots.pdf",
            "mera_mps_difference_vs_training_shots.pdf",
            "certificate_width_vs_certificate_shots.pdf",
        ]
        checks = {
            "analysis_complete": result["status"] == "complete",
            "all_expected_outputs_created": all((output / name).exists() for name in required),
            "paired_contrasts_nonempty": bool(result["paired_contrasts"]),
            "certificate_summary_nonempty": bool(result["certificate_summary"]),
        }
        print({"all_checks_passed": all(checks.values()), "checks": checks})
        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
