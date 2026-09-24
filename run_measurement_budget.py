#!/usr/bin/env python3
"""Run the frozen finite-measurement budget extension."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analyze_measurement_budget import analyze
from qipmera.budget import run_budget_case, terminal_certificate_samples
from qipmera.core import architecture, atomic_json, sha256_file, stable_seed, tfim_ground_state


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "measurement_budget_v1.json"
RESULTS = ROOT / "results" / "measurement_budget_v1"


def load_protocol() -> tuple[dict, str]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return protocol, sha256_file(PROTOCOL_PATH)


def cases(protocol: dict) -> list[dict]:
    return [
        {
            "n": int(protocol["system"]["n"]),
            "field": float(field),
            "replicate": int(replicate),
            "architecture": name,
        }
        for field in protocol["system"]["fields_h_over_J"]
        for replicate in protocol["system"]["evaluation_replicates"]
        for name in protocol["architectures"]
    ]


def case_name(case: dict) -> str:
    return (
        f"n{case['n']:02d}__h{case['field']:.2f}__rep{case['replicate']:02d}__"
        f"{case['architecture'].lower()}.json"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(protocol: dict, experiment_cases: list[dict]) -> dict:
    stages = protocol["training_stages"]
    maximum = sum(
        2 * int(stage["directions"]) * int(stage["steps"]) * int(stage["shots"])
        for stage in stages
    )
    expected_count = (
        len(protocol["system"]["fields_h_over_J"])
        * len(protocol["system"]["evaluation_replicates"])
        * len(protocol["architectures"])
    )
    specs = {name: architecture(protocol["system"]["n"], name) for name in protocol["architectures"]}
    checks = {
        "frozen_status": protocol["status"] == "FROZEN_MEASUREMENT_BUDGET_V1_BEFORE_RESULTS",
        "case_count": len(experiment_cases) == expected_count,
        "maximum_budget_560M": maximum == 560_000_000,
        "last_checkpoint_equals_maximum": max(protocol["requested_training_shot_checkpoints"]) == maximum,
        "both_models_225_coordinates": all(spec.coordinates == 225 for spec in specs.values()),
        "fresh_replicates_80_to_89": protocol["system"]["evaluation_replicates"] == list(range(80, 90)),
        "three_fixed_fields": protocol["system"]["fields_h_over_J"] == [0.9, 1.0, 1.1],
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "maximum_training_shots": maximum,
        "architecture_resources": {
            name: {
                "physical_su4_blocks": spec.physical_blocks,
                "independent_coordinates": spec.coordinates,
            }
            for name, spec in specs.items()
        },
    }


def worker(case: dict, protocol: dict, protocol_hash: str, output: str) -> dict:
    destination = Path(output) / "runs" / case_name(case)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("protocol_sha256") == protocol_hash and old.get("status") == "complete":
            return old
        raise RuntimeError(f"Existing result conflicts with frozen protocol: {destination}")

    result = run_budget_case(
        n=case["n"],
        field=case["field"],
        replicate=case["replicate"],
        architecture_name=case["architecture"],
        stages=protocol["training_stages"],
        requested_budgets=protocol["requested_training_shot_checkpoints"],
        cache_dir=Path(output) / "target_cache",
        seed_namespace=protocol["seed_policy"]["namespace"],
        seed_match_group=protocol["seed_policy"]["match_group"],
    )
    result["protocol_sha256"] = protocol_hash
    result["protocol_status"] = protocol["status"]
    atomic_json(destination, result)
    return result


def checkpoint_rows(results: list[dict], protocol: dict) -> list[dict]:
    entanglers_per_su4 = int(protocol["resource_accounting"]["entangling_primitives_per_su4"])
    rows = []
    for result in results:
        case = result["case"]
        blocks = result["architecture_resources"]["physical_su4_blocks"]
        for checkpoint in result["checkpoints"]:
            metric = checkpoint["evaluation"]
            achieved = checkpoint["achieved_training_shots"]
            rows.append(
                {
                    "n": case["n"],
                    "field": case["field_h_over_J"],
                    "replicate": case["replicate"],
                    "architecture": case["architecture"],
                    "requested_training_shots": checkpoint["requested_training_shots"],
                    "achieved_training_shots": achieved,
                    "objective_queries": checkpoint["objective_queries"],
                    "global_step_completed": checkpoint["global_step_completed"],
                    "fidelity": metric["fidelity"],
                    "zz_long_range_mae": metric["correlation_errors"]["zz_connected"]["long_range_mae"],
                    "energy_density_error": metric["energy_density_error"],
                    "half_chain_entropy_error": metric["half_chain_entropy_error"],
                    "p0": metric["p0"],
                    "joint_failure": metric["joint_failure"],
                    "eta_sum": metric["eta_sum"],
                    "success_F_gt_0.99": int(metric["fidelity"] > 0.99),
                    "physical_su4_blocks": blocks,
                    "independent_coordinates": result["architecture_resources"]["independent_coordinates"],
                    "abstract_su4_shot_gate_executions": achieved * blocks,
                    "modeled_entangling_shot_gate_executions": achieved * blocks * entanglers_per_su4,
                    "elapsed_seconds": checkpoint["elapsed_seconds"],
                }
            )
    return sorted(rows, key=lambda row: (row["field"], row["replicate"], row["architecture"], row["requested_training_shots"]))


def certificate_rows(results: list[dict], protocol: dict) -> list[dict]:
    cfg = protocol["certificate"]
    rows = []
    for result in results:
        case = result["case"]
        for checkpoint in result["checkpoints"]:
            metric = checkpoint["evaluation"]
            for shots in cfg["shots"]:
                seed = stable_seed(
                    "terminal_certificate",
                    case["field_h_over_J"],
                    case["replicate"],
                    case["architecture"],
                    checkpoint["requested_training_shots"],
                    shots,
                )
                samples = terminal_certificate_samples(
                    p0=metric["p0"],
                    infidelity=1.0 - metric["fidelity"],
                    certificate_shots=int(shots),
                    delta=float(cfg["delta"]),
                    tau=float(cfg["implementation_error_tau"]),
                    epsilon_top=float(cfg["epsilon_top"]),
                    repetitions=int(cfg["monte_carlo_repetitions"]),
                    seed=seed,
                )
                for sample in samples:
                    rows.append(
                        {
                            "field": case["field_h_over_J"],
                            "replicate": case["replicate"],
                            "architecture": case["architecture"],
                            "requested_training_shots": checkpoint["requested_training_shots"],
                            "exact_p0": metric["p0"],
                            "exact_infidelity": 1.0 - metric["fidelity"],
                            **sample,
                        }
                    )
    return rows


def execute(workers: int) -> dict:
    protocol, digest = load_protocol()
    experiment_cases = cases(protocol)
    plan = validate(protocol, experiment_cases)
    if not plan["all_checks_passed"]:
        raise RuntimeError(f"Frozen plan validation failed: {plan}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "started",
        "protocol_sha256": digest,
        "runner_sha256": sha256_file(Path(__file__)),
        "budget_module_sha256": sha256_file(ROOT / "qipmera" / "budget.py"),
        "core_sha256": sha256_file(ROOT / "qipmera" / "core.py"),
        "number_of_cases": len(experiment_cases),
        "plan_validation": plan,
        "cases": experiment_cases,
    }
    atomic_json(RESULTS / "execution_manifest.json", manifest)

    for n, field in sorted({(case["n"], case["field"]) for case in experiment_cases}):
        tfim_ground_state(n, field, RESULTS / "target_cache")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(worker, case, protocol, digest, str(RESULTS)): case
            for case in experiment_cases
        }
        for index, future in enumerate(as_completed(future_map), 1):
            result = future.result()
            results.append(result)
            final = result["checkpoints"][-1]["evaluation"]
            print(
                f"[{index}/{len(experiment_cases)}] {case_name(future_map[future])} "
                f"F={final['fidelity']:.7f} "
                f"ZZ-LR={final['correlation_errors']['zz_connected']['long_range_mae']:.7f}",
                flush=True,
            )

    run_rows = checkpoint_rows(results, protocol)
    cert_rows = certificate_rows(results, protocol)
    write_csv(RESULTS / "checkpoint_runs.csv", run_rows)
    write_csv(RESULTS / "certificate_samples.csv", cert_rows)
    report = {
        "status": "complete",
        "experiment": "measurement_budget_v1",
        "protocol_status": protocol["status"],
        "protocol_sha256": digest,
        "estimand": protocol["estimand"],
        "total_training_trajectories": len(results),
        "total_checkpoint_rows": len(run_rows),
        "checkpoint_runs": run_rows,
    }
    atomic_json(RESULTS / "report.json", report)
    manifest["status"] = "complete"
    atomic_json(RESULTS / "execution_manifest.json", manifest)
    analysis = analyze(RESULTS, PROTOCOL_PATH)
    return {"report": report, "analysis": analysis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    protocol, digest = load_protocol()
    experiment_cases = cases(protocol)
    if args.dry_run:
        print(json.dumps({
            "protocol_sha256": digest,
            "number_of_trajectories": len(experiment_cases),
            "plan_validation": validate(protocol, experiment_cases),
            "requested_checkpoints": protocol["requested_training_shot_checkpoints"],
            "certificate_shots": protocol["certificate"]["shots"],
        }, indent=2))
        return
    execute(args.workers)
    print(json.dumps({"status": "complete", "output": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
