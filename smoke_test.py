#!/usr/bin/env python3
"""Two short n=16 training cases that exercise the complete numerical path."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from qipmera.core import run_training_case, sha256_file


ROOT = Path(__file__).resolve().parent


def main() -> None:
    protocol_path = ROOT / "protocols" / "resource_frontier_v2.json"
    digest = sha256_file(protocol_path)
    stages = [
        {
            "steps": 2,
            "directions": 1,
            "shots": 100,
            "learning_rate": 0.02,
            "perturbation": 0.08,
        },
        {
            "steps": 1,
            "directions": 1,
            "shots": 200,
            "learning_rate": 0.008,
            "perturbation": 0.03,
        },
    ]
    checks = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = []
        for name in ("MERA_PM225", "MPS"):
            result = run_training_case(
                n=16,
                field=1.0,
                replicate=999,
                architecture_name=name,
                stages=stages,
                output_path=root / f"{name}.json",
                protocol_hash=digest,
                optimizer="global_spsa",
                cache_dir=root / "target_cache",
                seed_namespace="frontier_v2_smoke",
                seed_match_group="frontier_common_random_numbers",
            )
            results.append(result)
            evaluation = result["evaluation"]
            checks[f"{name}_complete"] = result["status"] == "complete"
            checks[f"{name}_finite_metrics"] = all(
                np.isfinite(value)
                for value in (
                    evaluation["fidelity"],
                    evaluation["eta_sum"],
                    evaluation["joint_failure"],
                    evaluation["energy_density_error"],
                    evaluation["half_chain_entropy_error"],
                    evaluation["correlation_errors"]["zz_connected"]["long_range_mae"],
                )
            )
            checks[f"{name}_resource_accounting"] = (
                result["training_resources"]["objective_queries"] == 6
                and result["training_resources"]["training_shots"] == 800
            )
        checks["common_random_number_seeds"] = results[0]["seeds"] == results[1]["seeds"]
        checks["checkpoint_files_written"] = all((root / f"{name}.json").exists() for name in ("MERA_PM225", "MPS"))

    checks = {key: bool(value) for key, value in checks.items()}
    output = {"all_checks_passed": all(checks.values()), "checks": checks}
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
