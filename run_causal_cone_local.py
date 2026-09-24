#!/usr/bin/env python3
"""Run the frozen causal-cone-complete local-circuit control."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from qipmera.core import architecture, atomic_json, run_training_case, sha256_file, stable_seed, tfim_ground_state


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "causal_cone_local_v1.json"
OUTPUT = ROOT / "results" / "causal_cone_local_v1"


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text()); return protocol, sha256_file(PROTOCOL_PATH)


def backward_cone(wire, blocks):
    cone = {wire}
    for left, right in reversed(blocks):
        if left in cone or right in cone: cone.update((left, right))
    return cone


def coverage_report(n, blocks, minimum_distance):
    uncovered = []
    for left in range(n):
        for right in range(left + minimum_distance, n):
            if backward_cone(left, blocks).isdisjoint(backward_cone(right, blocks)):
                uncovered.append([left, right])
    return {"all_pairs_covered": not uncovered, "uncovered_pairs": uncovered, "tested_pair_count": sum(n - distance for distance in range(minimum_distance, n))}


def brickwork_blocks(n, depth):
    return tuple((left, left + 1) for layer in range(depth) for left in range(layer % 2, n - 1, 2))


def validate(protocol):
    n = protocol["system"]["n"]; local = architecture(n, "LOCAL_LC225"); mera = architecture(n, "MERA_PM225")
    minimum_distance = min(protocol["system"]["held_out_distances"])
    coverage = coverage_report(n, local.blocks, minimum_distance)
    previous = coverage_report(n, brickwork_blocks(n, protocol["local_architecture"]["complete_brickwork_layers"] - 1), minimum_distance)
    checks = {
        "frozen_status": protocol["status"] == "FROZEN_CAUSAL_CONE_LOCAL_V1_BEFORE_RESULTS",
        "local_68_blocks": local.physical_blocks == 68,
        "both_225_coordinates": local.coordinates == mera.coordinates == 225,
        "all_held_out_pairs_covered": coverage["all_pairs_covered"],
        "nine_layers_are_minimal": not previous["all_pairs_covered"],
        "sharing_is_one_group_per_bond": local.parameter_groups == tuple(left for left, _ in local.blocks),
    }
    return {"all_checks_passed": all(checks.values()), "checks": checks, "coverage": coverage, "eight_layer_control": previous, "resources": {"MERA_PM225": {"blocks": mera.physical_blocks, "coordinates": mera.coordinates}, "LOCAL_LC225": {"blocks": local.physical_blocks, "coordinates": local.coordinates}}}


def cases(protocol):
    return [{"n": protocol["system"]["n"], "field": field, "replicate": replicate, "architecture": name} for field in protocol["system"]["fields_h_over_J"] for replicate in protocol["system"]["replicates"] for name in protocol["architectures"]]


def destination(output, case):
    return output / "runs" / f"n{case['n']:02d}__h{case['field']:.2f}__rep{case['replicate']:02d}__{case['architecture'].lower()}.json"


def worker(case, protocol, digest, output_text):
    output = Path(output_text)
    return run_training_case(
        n=case["n"], field=case["field"], replicate=case["replicate"], architecture_name=case["architecture"],
        stages=protocol["training_stages"], output_path=destination(output, case), protocol_hash=digest,
        optimizer="global_spsa", cache_dir=output / "target_cache", seed_namespace=protocol["seed_policy"]["namespace"],
        seed_match_group=protocol["seed_policy"]["match_group"],
    )


def interval(values, level):
    values = np.asarray(values, float); mean = float(values.mean()); se = float(values.std(ddof=1) / math.sqrt(len(values))); critical = float(t.ppf((1 + level) / 2, len(values) - 1)); return mean, se, mean - critical * se, mean + critical * se


def bootstrap(group, resamples, seed, level):
    rng = np.random.default_rng(seed); arrays = [group.loc[group.field == field, "difference"].to_numpy() for field in sorted(group.field.unique())]
    values = np.asarray([np.mean([rng.choice(a, len(a), replace=True).mean() for a in arrays]) for _ in range(resamples)]); alpha = 1 - level; return map(float, np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def analyze(results, protocol, output):
    rows = []
    for result in results:
        case = result["case"]; metric = result["evaluation"]; blocks = result["architecture_resources"]["physical_su4_blocks"]; shots = result["training_resources"]["training_shots"]
        rows.append({"field": case["field_h_over_J"], "replicate": case["replicate"], "architecture": case["architecture"], "fidelity": metric["fidelity"], "zz_long_range_mae": metric["correlation_errors"]["zz_connected"]["long_range_mae"], "energy_density_error": metric["energy_density_error"], "half_chain_entropy_error": metric["half_chain_entropy_error"], "joint_failure": metric["joint_failure"], "physical_su4_blocks": blocks, "independent_coordinates": result["architecture_resources"]["independent_coordinates"], "training_shots": shots, "su4_shot_gate_exposure": blocks * shots, "modeled_entangling_exposure": 3 * blocks * shots})
    frame = pd.DataFrame(rows).sort_values(["field", "replicate", "architecture"]); frame.to_csv(output / "runs.csv", index=False)
    summary = frame.groupby(["architecture", "field"], as_index=False).agg(count=("replicate", "size"), fidelity_mean=("fidelity", "mean"), fidelity_se=("fidelity", "sem"), zz_long_range_mae_mean=("zz_long_range_mae", "mean"), zz_long_range_mae_se=("zz_long_range_mae", "sem"), energy_density_error_mean=("energy_density_error", "mean"), half_chain_entropy_error_mean=("half_chain_entropy_error", "mean"), physical_su4_blocks=("physical_su4_blocks", "first"), su4_shot_gate_exposure=("su4_shot_gate_exposure", "first")); summary.to_csv(output / "summary.csv", index=False)
    level = protocol["statistics"]["interval_level"]; resamples = protocol["statistics"]["bootstrap_resamples"]; contrasts = []
    for metric in ["fidelity", "zz_long_range_mae", "energy_density_error", "half_chain_entropy_error"]:
        pivot = frame.pivot(index=["field", "replicate"], columns="architecture", values=metric).reset_index(); pivot["difference"] = pivot["MERA_PM225"] - pivot["LOCAL_LC225"]
        for field, group in pivot.groupby("field"):
            mean, se, low, high = interval(group.difference, level); contrasts.append({"metric": metric, "scope": "fixed_field", "field": float(field), "pairs": len(group), "mean_difference": mean, "standard_error": se, "interval_level": level, "interval_lower": low, "interval_upper": high})
        pooled = float(pivot.groupby("field").difference.mean().mean()); low, high = bootstrap(pivot, resamples, stable_seed("local_lc_bootstrap", metric), level); contrasts.append({"metric": metric, "scope": "equal_weight_fixed_fields", "field": None, "pairs": len(pivot), "mean_difference": pooled, "interval_level": level, "interval_lower": low, "interval_upper": high})
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    positions = np.arange(3); width = .36
    for axis, metric, ylabel in [(axes[0], "fidelity_mean", "Fidelity"), (axes[1], "zz_long_range_mae_mean", "Held-out long-range ZZ MAE")]:
        for offset, (name, label) in zip([-width/2, width/2], [("MERA_PM225", "MERA-225"), ("LOCAL_LC225", "Local-LC-225")]):
            values = summary[summary.architecture == name].sort_values("field"); error_col = metric.replace("_mean", "_se"); axis.bar(positions + offset, values[metric], width, yerr=values[error_col] if error_col in values else None, capsize=3, label=label)
        axis.set_xticks(positions, ["0.9", "1.0", "1.1"]); axis.set_xlabel("$h/J$"); axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.25)
    axes[-1].legend(frameon=False); fig.tight_layout(); fig.savefig(output / "causal_cone_local_summary.pdf", bbox_inches="tight"); fig.savefig(output / "causal_cone_local_summary.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    report = {"status": "complete", "experiment": "causal_cone_local_v1", "causal_cone_validation": validate(protocol), "contrasts": contrasts, "summaries": summary.to_dict(orient="records")}; atomic_json(output / "analysis.json", report); return report


def execute(workers):
    protocol, digest = load_protocol(); plan = validate(protocol)
    if not plan["all_checks_passed"]: raise RuntimeError(plan)
    experiment_cases = cases(protocol); OUTPUT.mkdir(parents=True, exist_ok=True); manifest = {"status": "started", "protocol_sha256": digest, "runner_sha256": sha256_file(Path(__file__)), "core_sha256": sha256_file(ROOT / "qipmera" / "core.py"), "number_of_cases": len(experiment_cases), "plan_validation": plan, "cases": experiment_cases}; atomic_json(OUTPUT / "execution_manifest.json", manifest)
    for n, field in sorted({(c["n"], c["field"]) for c in experiment_cases}): tfim_ground_state(n, field, OUTPUT / "target_cache")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, case, protocol, digest, str(OUTPUT)): case for case in experiment_cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result); case = futures[future]; print(f"[{index}/{len(experiment_cases)}] h={case['field']:.1f} rep={case['replicate']} {case['architecture']} F={result['evaluation']['fidelity']:.7f}", flush=True)
    report = analyze(results, protocol, OUTPUT); manifest["status"] = "complete"; atomic_json(OUTPUT / "execution_manifest.json", manifest); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1))); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args(); protocol, digest = load_protocol(); plan = validate(protocol)
    if args.dry_run: print(json.dumps({"protocol_sha256": digest, "number_of_cases": len(cases(protocol)), "plan_validation": plan}, indent=2)); return
    execute(args.workers)


if __name__ == "__main__": main()
