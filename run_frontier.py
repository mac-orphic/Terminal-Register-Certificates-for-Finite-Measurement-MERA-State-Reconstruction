#!/usr/bin/env python3
"""Run the frozen n=16 MERA/MPS resource-frontier experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analyze_frontier import analyze_report
from qipmera.core import (
    architecture,
    atomic_json,
    run_training_case,
    sha256_file,
    tfim_ground_state,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "resource_frontier_v2.json"
RESULTS = ROOT / "results"


def load_protocol() -> tuple[dict, str]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return protocol, sha256_file(PROTOCOL_PATH)


def build_cases(preset: str, protocol: dict) -> list[dict]:
    replicates = protocol["system"][f"{preset}_replicates"]
    return [
        {
            "n": int(protocol["system"]["n"]),
            "field": float(field),
            "replicate": int(replicate),
            "architecture": name,
            "stages": protocol["training_stages"],
        }
        for field in protocol["system"]["fields_h_over_J"]
        for replicate in replicates
        for name in protocol["architectures"]
    ]


def case_filename(case: dict) -> str:
    return (
        f"n{case['n']:02d}__h{case['field']:.2f}__rep{case['replicate']:02d}__"
        f"{case['architecture'].lower()}.json"
    )


def stage_resources(stages: list[dict]) -> tuple[int, int]:
    queries = sum(2 * int(s["directions"]) * int(s["steps"]) for s in stages)
    shots = sum(
        2 * int(s["directions"]) * int(s["steps"]) * int(s["shots"])
        for s in stages
    )
    return queries, shots


def worker(case: dict, output_dir: str, protocol_hash: str, namespace: str) -> dict:
    destination = Path(output_dir) / "runs" / case_filename(case)
    result = run_training_case(
        n=case["n"],
        field=case["field"],
        replicate=case["replicate"],
        architecture_name=case["architecture"],
        stages=case["stages"],
        output_path=destination,
        protocol_hash=protocol_hash,
        optimizer="global_spsa",
        cache_dir=Path(output_dir) / "target_cache",
        seed_namespace=namespace,
        seed_match_group="frontier_common_random_numbers",
    )
    atomic_json(destination, result)
    return result


def flatten(result: dict) -> dict:
    case = result["case"]
    evaluation = result["evaluation"]
    return {
        "n": case["n"],
        "field": case["field_h_over_J"],
        "replicate": case["replicate"],
        "architecture": case["architecture"],
        "fidelity": evaluation["fidelity"],
        "zz_long_range_mae": evaluation["correlation_errors"]["zz_connected"]["long_range_mae"],
        "energy_density_error": evaluation["energy_density_error"],
        "half_chain_entropy_error": evaluation["half_chain_entropy_error"],
        "eta_sum": evaluation["eta_sum"],
        "joint_failure": evaluation["joint_failure"],
        "physical_su4_blocks": result["architecture_resources"]["physical_su4_blocks"],
        "independent_coordinates": result["architecture_resources"]["independent_coordinates"],
        "objective_queries": result["training_resources"]["objective_queries"],
        "training_shots": result["training_resources"]["training_shots"],
        "base_seed": result["seeds"]["base"],
        "initialization_seed": result["seeds"]["initialization"],
        "measurement_seed": result["seeds"]["measurement"],
        "direction_seed": result["seeds"]["direction"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def grouped_summary(rows: list[dict]) -> list[dict]:
    output = []
    for architecture_name in sorted({r["architecture"] for r in rows}):
        for field in sorted({r["field"] for r in rows}):
            values = [r for r in rows if r["architecture"] == architecture_name and r["field"] == field]
            if not values:
                continue
            entry = {
                "architecture": architecture_name,
                "field": field,
                "count": len(values),
                "physical_su4_blocks": values[0]["physical_su4_blocks"],
                "independent_coordinates": values[0]["independent_coordinates"],
                "objective_queries": values[0]["objective_queries"],
                "training_shots": values[0]["training_shots"],
            }
            for metric in (
                "fidelity",
                "zz_long_range_mae",
                "energy_density_error",
                "half_chain_entropy_error",
                "eta_sum",
            ):
                data = np.asarray([r[metric] for r in values], dtype=float)
                entry[f"{metric}_mean"] = float(data.mean())
                entry[f"{metric}_se"] = (
                    float(data.std(ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0
                )
            output.append(entry)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(cases: list[dict], protocol: dict) -> dict:
    expected_resources = protocol["resource_table"]
    checks: dict[str, bool] = {}
    for name in protocol["architectures"]:
        spec = architecture(protocol["system"]["n"], name)
        checks[f"{name}_blocks"] = (
            spec.physical_blocks == expected_resources[name]["physical_su4_blocks"]
        )
        checks[f"{name}_coordinates"] = (
            spec.coordinates == expected_resources[name]["independent_coordinates"]
        )
    queries, shots = stage_resources(protocol["training_stages"])
    checks["objective_queries"] = queries == protocol["fixed_training_resources_per_run"]["objective_queries"]
    checks["training_shots"] = shots == protocol["fixed_training_resources_per_run"]["training_shots"]
    expected_cases = (
        len(protocol["system"]["fields_h_over_J"])
        * len({c["replicate"] for c in cases})
        * len(protocol["architectures"])
    )
    checks["case_count"] = len(cases) == expected_cases
    return {"all_checks_passed": all(checks.values()), "checks": checks}


def run(preset: str, workers: int, limit: int | None) -> dict:
    protocol, digest = load_protocol()
    cases = build_cases(preset, protocol)
    if limit is not None:
        cases = cases[:limit]
    plan_check = validate_plan(cases, protocol)
    if limit is None and not plan_check["all_checks_passed"]:
        raise RuntimeError(f"Frozen plan validation failed: {plan_check}")

    output = RESULTS / f"resource_frontier_{preset}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "started",
        "preset": preset,
        "protocol_sha256": digest,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(ROOT / "qipmera" / "core.py"),
        "protocol_status": protocol["status"],
        "number_of_cases": len(cases),
        "plan_validation": plan_check,
        "cases": cases,
    }
    atomic_json(output / "execution_manifest.json", manifest)

    # Ground states are generated serially to avoid cache races.
    for n, field in sorted({(c["n"], c["field"]) for c in cases}):
        tfim_ground_state(n, field, output / "target_cache")

    namespace = f"resource_frontier_v2_{preset}"
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker, case, str(output), digest, namespace): case for case in cases
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            evaluation = result["evaluation"]
            print(
                f"[{index}/{len(cases)}] {case_filename(futures[future])} "
                f"F={evaluation['fidelity']:.7f} "
                f"ZZ-LR={evaluation['correlation_errors']['zz_connected']['long_range_mae']:.7f}",
                flush=True,
            )

    rows = sorted(
        (flatten(result) for result in results),
        key=lambda r: (r["field"], r["replicate"], r["architecture"]),
    )
    summaries = grouped_summary(rows)
    report = {
        "status": "complete",
        "experiment": "resource_frontier_v2",
        "preset": preset,
        "protocol_sha256": digest,
        "protocol_status": protocol["status"],
        "total_runs": len(rows),
        "run_level_summary": rows,
        "group_summaries": summaries,
    }
    atomic_json(output / "report.json", report)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summaries)
    manifest["status"] = "complete"
    atomic_json(output / "execution_manifest.json", manifest)
    analysis = analyze_report(output / "report.json", PROTOCOL_PATH)
    return {"report": report, "analysis": analysis, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, help="Integration testing only; not a scientific run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    protocol, digest = load_protocol()
    cases = build_cases(args.preset, protocol)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.dry_run:
        queries, shots = stage_resources(protocol["training_stages"])
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "protocol_sha256": digest,
                    "cases": len(cases),
                    "architectures": protocol["architectures"],
                    "objective_queries_per_case": queries,
                    "training_shots_per_case": shots,
                    "total_training_shots": shots * len(cases),
                    "plan_validation": validate_plan(cases, protocol),
                },
                indent=2,
            )
        )
        return

    result = run(args.preset, args.workers, args.limit)
    print(json.dumps({"status": "complete", "output": result["output"]}, indent=2))


if __name__ == "__main__":
    main()
