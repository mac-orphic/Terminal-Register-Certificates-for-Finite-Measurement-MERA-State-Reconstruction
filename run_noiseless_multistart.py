#!/usr/bin/env python3
"""Run the frozen exact-autodiff multistart capacity diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qipmera.core import architecture, atomic_json, sha256_file, tfim_ground_state
from qipmera.exact_torch import require_torch, train_exact_adam


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "noiseless_multistart_v1.json"
OUTPUT = ROOT / "results" / "noiseless_multistart_v1"


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return protocol, sha256_file(PROTOCOL_PATH)


def build_cases(protocol):
    return [
        {"n": protocol["system"]["n"], "field": field, "restart": restart, "architecture": name}
        for field in protocol["system"]["fields_h_over_J"]
        for restart in protocol["system"]["restart_seeds"]
        for name in protocol["architectures"]
    ]


def result_path(output, case):
    return output / "runs" / f"n{case['n']:02d}__h{case['field']:.2f}__restart{case['restart']:03d}__{case['architecture'].lower()}.json"


def validate(protocol):
    specs = {name: architecture(protocol["system"]["n"], name) for name in protocol["architectures"]}
    schedule = protocol["optimizer"]["learning_rate_schedule"]
    checks = {
        "frozen_status": protocol["status"] == "FROZEN_NOISELESS_MULTISTART_V1_BEFORE_RESULTS",
        "twenty_restarts": len(protocol["system"]["restart_seeds"]) == 20 and len(set(protocol["system"]["restart_seeds"])) == 20,
        "equal_225_coordinates": all(spec.coordinates == 225 for spec in specs.values()),
        "five_thousand_steps": protocol["optimizer"]["maximum_steps"] == 5000,
        "complete_lr_schedule": schedule[0]["start_step"] == 0 and schedule[-1]["end_step"] == 4999,
        "exact_objective": protocol["objective"].startswith("exact eta_sum"),
    }
    return {"all_checks_passed": all(checks.values()), "checks": checks, "architecture_resources": {name: {"physical_su4_blocks": spec.physical_blocks, "independent_coordinates": spec.coordinates} for name, spec in specs.items()}}


def worker(case, protocol, digest, output_text, device):
    output = Path(output_text); destination = result_path(output, case)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("protocol_sha256") == digest and old.get("status") == "complete": return old
        raise RuntimeError(f"Conflicting existing result: {destination}")
    target, energy = tfim_ground_state(case["n"], case["field"], output / "target_cache")
    spec = architecture(case["n"], case["architecture"])
    result = train_exact_adam(
        target_numpy=target, ground_energy=energy, n=case["n"], field=case["field"], spec=spec,
        restart=case["restart"], namespace=protocol["seed_policy"]["namespace"],
        match_group=protocol["seed_policy"]["match_group"], optimizer_config=protocol["optimizer"], device_name=device,
    )
    result.update({"protocol_sha256": digest, "case": case, "architecture_resources": {"physical_su4_blocks": spec.physical_blocks, "independent_coordinates": spec.coordinates}})
    atomic_json(destination, result); return result


def analyze(results, protocol, output):
    rows = []
    for result in results:
        metric = result["evaluation"]; case = result["case"]
        rows.append({
            "field": case["field"], "restart": case["restart"], "architecture": case["architecture"],
            "converged": result["converged"], "completed_steps": result["completed_steps"],
            "final_gradient_norm": result["final_gradient_norm"], "elapsed_seconds": result["elapsed_seconds"],
            "fidelity": metric["fidelity"], "zz_long_range_mae": metric["correlation_errors"]["zz_connected"]["long_range_mae"],
            "energy_density_error": metric["energy_density_error"], "half_chain_entropy_error": metric["half_chain_entropy_error"],
            "joint_failure": metric["joint_failure"], "eta_sum": metric["eta_sum"],
        })
    frame = pd.DataFrame(rows).sort_values(["field", "restart", "architecture"]); frame.to_csv(output / "runs.csv", index=False)
    summaries = []
    for (architecture_name, field), group in frame.groupby(["architecture", "field"]):
        row = {"architecture": architecture_name, "field": float(field), "restarts": len(group), "converged_count": int(group.converged.sum()), "convergence_fraction": float(group.converged.mean())}
        for metric in ["fidelity", "zz_long_range_mae", "eta_sum", "joint_failure"]:
            values = group[metric]
            row.update({f"{metric}_best": float(values.max() if metric == "fidelity" else values.min()), f"{metric}_median": float(values.median()), f"{metric}_worst": float(values.min() if metric == "fidelity" else values.max())})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)

    contrasts = []
    for metric in ["fidelity", "zz_long_range_mae", "eta_sum"]:
        pivot = frame.pivot(index=["field", "restart"], columns="architecture", values=metric).reset_index()
        pivot["difference"] = pivot["MERA_PM225"] - pivot["MPS"]
        for field, group in pivot.groupby("field"):
            contrasts.append({"metric": metric, "field": float(field), "pairs": len(group), "difference_definition": "MERA_PM225 minus MPS", "mean_difference": float(group.difference.mean()), "median_difference": float(group.difference.median()), "mera_win_rate": float((group.difference > 0).mean() if metric == "fidelity" else (group.difference < 0).mean())})
    pd.DataFrame(contrasts).to_csv(output / "paired_restart_contrasts.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for axis, metric, label in [(axes[0], "fidelity", "Noiseless fidelity"), (axes[1], "zz_long_range_mae", "Held-out long-range ZZ MAE")]:
        data = []; labels = []
        for field in sorted(frame.field.unique()):
            for architecture_name, short in [("MERA_PM225", "MERA"), ("MPS", "MPS")]:
                data.append(frame[(frame.field == field) & (frame.architecture == architecture_name)][metric].to_numpy()); labels.append(f"{short}\n{field:.1f}")
        axis.boxplot(data, labels=labels, showfliers=True); axis.set_ylabel(label); axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "noiseless_multistart_summary.pdf", bbox_inches="tight"); fig.savefig(output / "noiseless_multistart_summary.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    report = {"status": "complete", "experiment": "noiseless_multistart_v1", "warning": "Best of 20 is best-found, not a certified global optimum.", "summaries": summaries, "paired_restart_contrasts": contrasts}
    atomic_json(output / "analysis.json", report); return report


def execute(workers, device):
    torch = require_torch(); protocol, digest = load_protocol(); plan = validate(protocol)
    if not plan["all_checks_passed"]: raise RuntimeError(plan)
    resolved_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if resolved_cuda and workers != 1: raise ValueError("Use --workers 1 when CUDA is selected explicitly or by --device auto.")
    cases = build_cases(protocol); OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "started", "protocol_sha256": digest, "runner_sha256": sha256_file(Path(__file__)), "exact_backend_sha256": sha256_file(ROOT / "qipmera" / "exact_torch.py"), "number_of_cases": len(cases), "plan_validation": plan, "device": device, "cases": cases}
    atomic_json(OUTPUT / "execution_manifest.json", manifest)
    for n, field in sorted({(c["n"], c["field"]) for c in cases}): tfim_ground_state(n, field, OUTPUT / "target_cache")
    results = []
    if workers == 1:
        for index, case in enumerate(cases, 1):
            result = worker(case, protocol, digest, str(OUTPUT), device); results.append(result); print(f"[{index}/{len(cases)}] h={case['field']:.1f} restart={case['restart']} {case['architecture']} F={result['evaluation']['fidelity']:.7f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, case, protocol, digest, str(OUTPUT), device): case for case in cases}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result(); results.append(result); case = futures[future]; print(f"[{index}/{len(cases)}] h={case['field']:.1f} restart={case['restart']} {case['architecture']} F={result['evaluation']['fidelity']:.7f}", flush=True)
    report = analyze(results, protocol, OUTPUT); manifest["status"] = "complete"; atomic_json(OUTPUT / "execution_manifest.json", manifest); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto"); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    protocol, digest = load_protocol(); plan = validate(protocol)
    if args.dry_run:
        try:
            torch = require_torch()
            dependency = {"torch_available": True, "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
        except RuntimeError:
            dependency = {"torch_available": False, "installation_required_before_execution": True}
        print(json.dumps({"protocol_sha256": digest, "number_of_cases": len(build_cases(protocol)), "plan_validation": plan, **dependency}, indent=2)); return
    execute(args.workers, args.device)


if __name__ == "__main__": main()
