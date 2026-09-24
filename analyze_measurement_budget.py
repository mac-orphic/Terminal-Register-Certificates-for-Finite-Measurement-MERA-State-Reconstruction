#!/usr/bin/env python3
"""Analyze and plot the frozen measurement-budget experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, t

from qipmera.core import atomic_json, stable_seed


def mean_interval(values: np.ndarray, level: float) -> tuple[float, float, float, float]:
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0, mean, mean
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(t.ppf((1.0 + level) / 2.0, len(values) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def clopper_pearson(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def grouped_summaries(frame: pd.DataFrame) -> list[dict]:
    output = []
    keys = ["architecture", "field", "requested_training_shots"]
    for (architecture, field, budget), group in frame.groupby(keys, sort=True):
        successes = int(group["success_F_gt_0.99"].sum())
        lower, upper = clopper_pearson(successes, len(group))
        row = {
            "architecture": architecture,
            "field": float(field),
            "requested_training_shots": int(budget),
            "achieved_training_shots": int(group["achieved_training_shots"].iloc[0]),
            "count": int(len(group)),
            "successes_F_gt_0.99": successes,
            "success_probability": successes / len(group),
            "success_cp95_lower": lower,
            "success_cp95_upper": upper,
        }
        for metric in ["fidelity", "zz_long_range_mae", "energy_density_error", "half_chain_entropy_error", "joint_failure"]:
            mean, se, low, high = mean_interval(group[metric].to_numpy(), 0.95)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_lower"] = low
            row[f"{metric}_ci95_upper"] = high
        output.append(row)
    return output


def paired_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["field", "replicate", "requested_training_shots"],
        columns="architecture",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if pivot[["MERA_PM225", "MPS"]].isna().any().any():
        raise RuntimeError(f"Incomplete MERA/MPS pairing for {metric}.")
    pivot["difference"] = pivot["MERA_PM225"] - pivot["MPS"]
    return pivot


def stratified_bootstrap(paired: pd.DataFrame, resamples: int, seed: int, level: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fields = sorted(paired["field"].unique())
    draws = np.empty(resamples)
    arrays = [paired.loc[paired["field"] == field, "difference"].to_numpy() for field in fields]
    for index in range(resamples):
        draws[index] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean()
            for values in arrays
        ])
    alpha = 1.0 - level
    return tuple(map(float, np.quantile(draws, [alpha / 2, 1 - alpha / 2])))


def contrasts(frame: pd.DataFrame, protocol: dict) -> list[dict]:
    level = float(protocol["statistics"]["confirmatory_interval_level"])
    resamples = int(protocol["statistics"]["bootstrap_resamples"])
    output = []
    for metric in ["zz_long_range_mae", "fidelity", "energy_density_error", "half_chain_entropy_error"]:
        paired = paired_frame(frame, metric)
        for budget, budget_group in paired.groupby("requested_training_shots", sort=True):
            for field, field_group in budget_group.groupby("field", sort=True):
                mean, se, low, high = mean_interval(field_group["difference"].to_numpy(), level)
                output.append({
                    "metric": metric,
                    "scope": "fixed_field",
                    "field": float(field),
                    "requested_training_shots": int(budget),
                    "pairs": int(len(field_group)),
                    "difference_definition": "MERA_PM225 minus MPS",
                    "mean_difference": mean,
                    "standard_error": se,
                    "interval_level": level,
                    "t_interval_lower": low,
                    "t_interval_upper": high,
                    "mera_win_rate": float((field_group["difference"] < 0).mean()) if "error" in metric or "mae" in metric else float((field_group["difference"] > 0).mean()),
                })

            field_means = budget_group.groupby("field")["difference"].mean()
            equal_field_mean = float(field_means.mean())
            boot_low, boot_high = stratified_bootstrap(
                budget_group,
                resamples,
                stable_seed("measurement_budget_bootstrap", metric, int(budget)),
                level,
            )
            output.append({
                "metric": metric,
                "scope": "equal_weight_three_fixed_fields",
                "field": None,
                "requested_training_shots": int(budget),
                "pairs": int(len(budget_group)),
                "difference_definition": "MERA_PM225 minus MPS",
                "mean_difference": equal_field_mean,
                "interval_level": level,
                "stratified_bootstrap_lower": boot_low,
                "stratified_bootstrap_upper": boot_high,
                "estimand_warning": "Fixed-field equal-weight optimizer-seed estimand; not arbitrary-field generalization.",
            })
    return output


def certificate_summary(certificate: pd.DataFrame) -> list[dict]:
    keys = ["architecture", "requested_training_shots", "certificate_shots"]
    output = []
    for (architecture, training_budget, certificate_shots), group in certificate.groupby(keys, sort=True):
        coverage = float(group["covers_exact_infidelity"].mean())
        output.append({
            "architecture": architecture,
            "requested_training_shots": int(training_budget),
            "certificate_shots": int(certificate_shots),
            "sample_count": int(len(group)),
            "hoeffding_radius": float(group["hoeffding_radius"].iloc[0]),
            "mean_certificate_upper_bound": float(group["certificate_upper_bound"].mean()),
            "empirical_coverage": coverage,
            "empirical_failure_rate": 1.0 - coverage,
        })
    return output


def plots(frame: pd.DataFrame, summaries: pd.DataFrame, contrast_rows: list[dict], certificate: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })
    labels = {"MERA_PM225": "MERA-225", "MPS": "MPS-225"}
    colors = {"MERA_PM225": "#2878B5", "MPS": "#D95319"}

    for metric, ylabel, filename in [
        ("fidelity_mean", "Reconstruction fidelity", "fidelity_vs_training_shots"),
        ("zz_long_range_mae_mean", "Held-out long-range ZZ MAE", "long_range_error_vs_training_shots"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
        for axis, field in zip(axes, sorted(summaries["field"].unique())):
            subset = summaries[summaries["field"] == field]
            for architecture in ["MERA_PM225", "MPS"]:
                values = subset[subset["architecture"] == architecture].sort_values("requested_training_shots")
                axis.errorbar(values["requested_training_shots"], values[metric], yerr=values[metric.replace("_mean", "_se")], marker="o", capsize=3, label=labels[architecture], color=colors[architecture])
            axis.set_xscale("log")
            axis.set_title(f"$h/J={field:.1f}$")
            axis.set_xlabel("Training shots")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for axis, field in zip(axes, sorted(summaries["field"].unique())):
        subset = summaries[summaries["field"] == field]
        for architecture in ["MERA_PM225", "MPS"]:
            values = subset[subset["architecture"] == architecture].sort_values("requested_training_shots")
            yerr = np.vstack([
                values["success_probability"] - values["success_cp95_lower"],
                values["success_cp95_upper"] - values["success_probability"],
            ])
            axis.errorbar(values["requested_training_shots"], values["success_probability"], yerr=yerr, marker="o", capsize=3, label=labels[architecture], color=colors[architecture])
        axis.set_xscale("log"); axis.set_ylim(-0.03, 1.03)
        axis.set_title(f"$h/J={field:.1f}$"); axis.set_xlabel("Training shots"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("Success probability ($F>0.99$)")
    axes[-1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "success_probability_vs_training_shots.pdf", bbox_inches="tight"); fig.savefig(output / "success_probability_vs_training_shots.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    pooled = pd.DataFrame([row for row in contrast_rows if row["metric"] == "zz_long_range_mae" and row["scope"] == "equal_weight_three_fixed_fields"]).sort_values("requested_training_shots")
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.errorbar(pooled["requested_training_shots"], pooled["mean_difference"], yerr=np.vstack([pooled["mean_difference"] - pooled["stratified_bootstrap_lower"], pooled["stratified_bootstrap_upper"] - pooled["mean_difference"]]), marker="o", capsize=4, color="#2878B5")
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xscale("log"); axis.set_xlabel("Training shots"); axis.set_ylabel(r"$\Delta E_{\rm LR}$ (MERA-225 $-$ MPS-225)"); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "mera_mps_difference_vs_training_shots.pdf", bbox_inches="tight"); fig.savefig(output / "mera_mps_difference_vs_training_shots.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    radius = certificate.groupby("certificate_shots", as_index=False)["hoeffding_radius"].first().sort_values("certificate_shots")
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(radius["certificate_shots"], radius["hoeffding_radius"], marker="o", color="#3BA272")
    axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("Independent certificate shots"); axis.set_ylabel("Hoeffding correction radius"); axis.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(output / "certificate_width_vs_certificate_shots.pdf", bbox_inches="tight"); fig.savefig(output / "certificate_width_vs_certificate_shots.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def analyze(results_dir: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    frame = pd.read_csv(results_dir / "checkpoint_runs.csv")
    certificate = pd.read_csv(results_dir / "certificate_samples.csv")
    summaries = grouped_summaries(frame)
    contrast_rows = contrasts(frame, protocol)
    cert_summary = certificate_summary(certificate)
    pd.DataFrame(summaries).to_csv(results_dir / "budget_summary.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(results_dir / "paired_contrasts.csv", index=False)
    pd.DataFrame(cert_summary).to_csv(results_dir / "certificate_summary.csv", index=False)
    resource_audit = frame.groupby("architecture", as_index=False).agg(
        physical_su4_blocks=("physical_su4_blocks", "first"),
        independent_coordinates=("independent_coordinates", "first"),
        max_training_shots=("achieved_training_shots", "max"),
        max_abstract_su4_shot_gate_executions=("abstract_su4_shot_gate_executions", "max"),
        max_modeled_entangling_shot_gate_executions=("modeled_entangling_shot_gate_executions", "max"),
    )
    resource_audit.to_csv(results_dir / "resource_audit.csv", index=False)
    plots(frame, pd.DataFrame(summaries), contrast_rows, certificate, results_dir)
    result = {
        "status": "complete",
        "estimand": protocol["estimand"],
        "confirmatory_interval_level": protocol["statistics"]["confirmatory_interval_level"],
        "group_summaries": summaries,
        "paired_contrasts": contrast_rows,
        "certificate_summary": cert_summary,
        "resource_audit": resource_audit.to_dict(orient="records"),
    }
    atomic_json(results_dir / "analysis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results/measurement_budget_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("protocols/measurement_budget_v1.json"))
    args = parser.parse_args()
    analyze(args.results_dir, args.protocol)


if __name__ == "__main__":
    main()
