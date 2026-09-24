#!/usr/bin/env python3
"""Run the frozen equal aggregate gate-exposure comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from qipmera.budget import run_budget_case
from qipmera.core import architecture, atomic_json, sha256_file, stable_seed, tfim_ground_state


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "gate_exposure_v1.json"
OUTPUT = ROOT / "results" / "gate_exposure_v1"


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return protocol, sha256_file(PROTOCOL_PATH)


def build_cases(protocol):
    return [
        {"n": protocol["system"]["n"], "field": field, "replicate": replicate, "architecture": name}
        for field in protocol["system"]["fields_h_over_J"]
        for replicate in protocol["system"]["replicates"]
        for name in protocol["architectures"]
    ]


def case_path(output: Path, case: dict) -> Path:
    return output / "runs" / (
        f"n{case['n']:02d}__h{case['field']:.2f}__rep{case['replicate']:02d}__"
        f"{case['architecture'].lower()}.json"
    )


def requested_training_budgets(protocol, architecture_name):
    blocks = architecture(protocol["system"]["n"], architecture_name).physical_blocks
    return [int(math.ceil(target / blocks)) for target in protocol["common_su4_shot_gate_targets"]]


def validate(protocol):
    specs = {name: architecture(protocol["system"]["n"], name) for name in protocol["architectures"]}
    maximum = sum(2 * s["directions"] * s["steps"] * s["shots"] for s in protocol["training_stages"])
    requested = {name: requested_training_budgets(protocol, name) for name in specs}
    checks = {
        "frozen_status": protocol["status"] == "FROZEN_GATE_EXPOSURE_V1_BEFORE_RESULTS",
        "equal_coordinates": all(spec.coordinates == 225 for spec in specs.values()),
        "expected_blocks": specs["MERA_PM225"].physical_blocks == 26 and specs["MPS"].physical_blocks == 15,
        "all_checkpoints_within_560M": all(max(values) <= maximum for values in requested.values()),
        "same_frozen_seed_namespace": protocol["seed_policy"]["namespace"] == "measurement_budget_v1_evaluation",
    }
    return {"all_checks_passed": all(checks.values()), "checks": checks, "requested_training_shots": requested, "maximum_training_shots": maximum}


def worker(case, protocol, digest, output_text):
    output = Path(output_text)
    destination = case_path(output, case)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("protocol_sha256") == digest and old.get("status") == "complete":
            return old
        raise RuntimeError(f"Conflicting existing result: {destination}")
    requested = requested_training_budgets(protocol, case["architecture"])
    result = run_budget_case(
        n=case["n"], field=case["field"], replicate=case["replicate"],
        architecture_name=case["architecture"], stages=protocol["training_stages"],
        requested_budgets=requested, cache_dir=output / "target_cache",
        seed_namespace=protocol["seed_policy"]["namespace"],
        seed_match_group=protocol["seed_policy"]["match_group"],
    )
    result["protocol_sha256"] = digest
    targets = protocol["common_su4_shot_gate_targets"]
    blocks = result["architecture_resources"]["physical_su4_blocks"]
    for target, checkpoint in zip(targets, result["checkpoints"]):
        checkpoint["target_su4_shot_gate_exposure"] = int(target)
        checkpoint["achieved_su4_shot_gate_exposure"] = int(checkpoint["achieved_training_shots"] * blocks)
        checkpoint["achieved_modeled_entangling_exposure"] = int(3 * checkpoint["achieved_training_shots"] * blocks)
    atomic_json(destination, result)
    return result


def flatten(results):
    rows = []
    for result in results:
        case = result["case"]
        for checkpoint in result["checkpoints"]:
            metric = checkpoint["evaluation"]
            rows.append({
                "field": case["field_h_over_J"], "replicate": case["replicate"],
                "architecture": case["architecture"],
                "target_su4_shot_gate_exposure": checkpoint["target_su4_shot_gate_exposure"],
                "achieved_su4_shot_gate_exposure": checkpoint["achieved_su4_shot_gate_exposure"],
                "achieved_modeled_entangling_exposure": checkpoint["achieved_modeled_entangling_exposure"],
                "achieved_training_shots": checkpoint["achieved_training_shots"],
                "objective_queries": checkpoint["objective_queries"],
                "fidelity": metric["fidelity"],
                "zz_long_range_mae": metric["correlation_errors"]["zz_connected"]["long_range_mae"],
                "energy_density_error": metric["energy_density_error"],
                "half_chain_entropy_error": metric["half_chain_entropy_error"],
                "joint_failure": metric["joint_failure"],
                "physical_su4_blocks": result["architecture_resources"]["physical_su4_blocks"],
                "independent_coordinates": result["architecture_resources"]["independent_coordinates"],
            })
    return sorted(rows, key=lambda row: (row["target_su4_shot_gate_exposure"], row["field"], row["replicate"], row["architecture"]))


def interval(values, level):
    values = np.asarray(values, float); mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(t.ppf((1 + level) / 2, len(values) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def stratified_bootstrap(paired, resamples, seed, level):
    rng = np.random.default_rng(seed); fields = sorted(paired.field.unique())
    arrays = [paired.loc[paired.field == field, "difference"].to_numpy() for field in fields]
    draws = np.asarray([np.mean([rng.choice(a, len(a), replace=True).mean() for a in arrays]) for _ in range(resamples)])
    alpha = 1 - level
    return map(float, np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def analyze(rows, protocol, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "runs.csv", index=False)
    metrics = ["fidelity", "zz_long_range_mae", "energy_density_error", "half_chain_entropy_error"]
    level = protocol["statistics"]["interval_level"]
    resamples = protocol["statistics"]["bootstrap_resamples"]
    summaries = frame.groupby(["architecture", "field", "target_su4_shot_gate_exposure"], as_index=False).agg(
        count=("replicate", "size"), achieved_training_shots=("achieved_training_shots", "first"),
        achieved_su4_shot_gate_exposure=("achieved_su4_shot_gate_exposure", "first"),
        fidelity_mean=("fidelity", "mean"), fidelity_se=("fidelity", "sem"),
        zz_long_range_mae_mean=("zz_long_range_mae", "mean"), zz_long_range_mae_se=("zz_long_range_mae", "sem"),
        energy_density_error_mean=("energy_density_error", "mean"),
        half_chain_entropy_error_mean=("half_chain_entropy_error", "mean"),
    )
    summaries.to_csv(output / "summary.csv", index=False)
    contrasts = []
    for metric in metrics:
        pivot = frame.pivot(index=["field", "replicate", "target_su4_shot_gate_exposure"], columns="architecture", values=metric).reset_index()
        pivot["difference"] = pivot["MERA_PM225"] - pivot["MPS"]
        for target, target_group in pivot.groupby("target_su4_shot_gate_exposure"):
            for field, group in target_group.groupby("field"):
                mean, se, low, high = interval(group.difference, level)
                contrasts.append({"metric": metric, "scope": "fixed_field", "field": float(field), "target_su4_shot_gate_exposure": int(target), "pairs": len(group), "mean_difference": mean, "standard_error": se, "interval_level": level, "interval_lower": low, "interval_upper": high})
            pooled_mean = float(target_group.groupby("field").difference.mean().mean())
            low, high = stratified_bootstrap(target_group, resamples, stable_seed("gate_exposure_bootstrap", metric, int(target)), level)
            contrasts.append({"metric": metric, "scope": "equal_weight_fixed_fields", "field": None, "target_su4_shot_gate_exposure": int(target), "pairs": len(target_group), "mean_difference": pooled_mean, "interval_level": level, "interval_lower": low, "interval_upper": high})
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    for metric, ylabel, name in [("fidelity_mean", "Fidelity", "fidelity_vs_gate_exposure"), ("zz_long_range_mae_mean", "Held-out long-range ZZ MAE", "long_range_error_vs_gate_exposure")]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for axis, field in zip(axes, sorted(summaries.field.unique())):
            subset = summaries[summaries.field == field]
            for architecture, label in [("MERA_PM225", "MERA-225"), ("MPS", "MPS-225")]:
                values = subset[subset.architecture == architecture].sort_values("target_su4_shot_gate_exposure")
                se_col = metric.replace("_mean", "_se")
                yerr = values[se_col] if se_col in values else None
                axis.errorbar(values.target_su4_shot_gate_exposure, values[metric], yerr=yerr, marker="o", capsize=3, label=label)
            axis.set_xscale("log"); axis.set_title(f"$h/J={field:.1f}$"); axis.set_xlabel("SU(4)-shot-gate exposure"); axis.grid(alpha=.25)
        axes[0].set_ylabel(ylabel); axes[-1].legend(frameon=False); fig.tight_layout()
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight"); fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    report = {"status": "complete", "experiment": "gate_exposure_v1", "estimand": protocol["statistics"]["pooled_estimand"], "contrasts": contrasts, "resource_rows": summaries.to_dict(orient="records")}
    atomic_json(output / "analysis.json", report)
    return report


def execute(workers):
    protocol, digest = load_protocol(); plan = validate(protocol)
    if not plan["all_checks_passed"]: raise RuntimeError(plan)
    cases = build_cases(protocol); OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "started", "protocol_sha256": digest, "runner_sha256": sha256_file(Path(__file__)), "number_of_cases": len(cases), "plan_validation": plan, "cases": cases}
    atomic_json(OUTPUT / "execution_manifest.json", manifest)
    for n, field in sorted({(c["n"], c["field"]) for c in cases}): tfim_ground_state(n, field, OUTPUT / "target_cache")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, case, protocol, digest, str(OUTPUT)): case for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result)
            print(f"[{index}/{len(cases)}] {future_to_name(futures[future])}", flush=True)
    report = analyze(flatten(results), protocol, OUTPUT)
    manifest["status"] = "complete"; atomic_json(OUTPUT / "execution_manifest.json", manifest)
    return report


def future_to_name(case):
    return f"h={case['field']:.1f} rep={case['replicate']} {case['architecture']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1))); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    protocol, digest = load_protocol(); plan = validate(protocol)
    if args.dry_run:
        print(json.dumps({"protocol_sha256": digest, "number_of_cases": len(build_cases(protocol)), "plan_validation": plan}, indent=2)); return
    execute(args.workers)


if __name__ == "__main__": main()
