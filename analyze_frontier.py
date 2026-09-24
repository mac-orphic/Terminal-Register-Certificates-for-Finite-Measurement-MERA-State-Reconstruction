#!/usr/bin/env python3
"""Create paired contrasts, decisions, and plots for resource-frontier v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from qipmera.core import atomic_json, stable_seed


METRICS = (
    "fidelity",
    "zz_long_range_mae",
    "energy_density_error",
    "half_chain_entropy_error",
    "eta_sum",
)


def t_interval(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


def bootstrap_interval(values: np.ndarray, label: str, resamples: int = 20000) -> list[float]:
    rng = np.random.default_rng(stable_seed("frontier_bootstrap", label))
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def make_contrast(
    rows: list[dict], name: str, challenger: str, baseline: str, scope: str
) -> dict:
    lookup = {(r["field"], r["replicate"], r["architecture"]): r for r in rows}
    paired_keys = sorted(
        {
            (r["field"], r["replicate"])
            for r in rows
            if (r["field"], r["replicate"], challenger) in lookup
            and (r["field"], r["replicate"], baseline) in lookup
        }
    )
    result = {
        "name": name,
        "scope": scope,
        "challenger": challenger,
        "baseline": baseline,
        "pairs": len(paired_keys),
        "metrics": {},
        "per_field": [],
    }
    for metric in METRICS:
        differences = np.asarray(
            [
                lookup[(field, replicate, challenger)][metric]
                - lookup[(field, replicate, baseline)][metric]
                for field, replicate in paired_keys
            ],
            dtype=float,
        )
        favors = differences > 0 if metric == "fidelity" else differences < 0
        result["metrics"][metric] = {
            "mean_difference": float(differences.mean()),
            "standard_error": float(stats.sem(differences)) if len(differences) > 1 else 0.0,
            "median_difference": float(np.median(differences)),
            "minimum_difference": float(differences.min()),
            "maximum_difference": float(differences.max()),
            "t95_interval": t_interval(differences),
            "paired_bootstrap_95_interval": bootstrap_interval(differences, f"{name}_{metric}"),
            "challenger_win_rate": float(np.mean(favors)),
            "sign_convention": (
                "positive favors challenger" if metric == "fidelity" else "negative favors challenger"
            ),
        }
    for field in sorted({key[0] for key in paired_keys}):
        keys = [key for key in paired_keys if key[0] == field]
        field_entry = {"field": field, "pairs": len(keys), "metrics": {}}
        for metric in METRICS:
            differences = np.asarray(
                [
                    lookup[(f, replicate, challenger)][metric]
                    - lookup[(f, replicate, baseline)][metric]
                    for f, replicate in keys
                ],
                dtype=float,
            )
            field_entry["metrics"][metric] = {
                "mean_difference": float(differences.mean()),
                "t95_interval": t_interval(differences),
            }
        result["per_field"].append(field_entry)
    return result


def plot_frontier(rows: list[dict], destination: Path) -> None:
    architectures = [
        "MERA",
        "MERA_PM225",
        "MPS",
        "TTN",
        "BRICKWORK225",
        "BRICKWORK390",
    ]
    colors = {
        "MERA": "#2166ac",
        "MERA_PM225": "#67a9cf",
        "MPS": "#1b9e77",
        "TTN": "#d95f02",
        "BRICKWORK225": "#7570b3",
        "BRICKWORK390": "#e7298a",
    }
    labels = {
        "MERA": "MERA 390",
        "MERA_PM225": "MERA 225",
        "MPS": "MPS 225",
        "TTN": "TTN 225",
        "BRICKWORK225": "Local 225",
        "BRICKWORK390": "Local 390",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, title in (
        (axes[0], "fidelity", "Reconstruction fidelity"),
        (axes[1], "zz_long_range_mae", "Held-out long-range ZZ error"),
    ):
        means = []
        errors = []
        for name in architectures:
            values = np.asarray([r[metric] for r in rows if r["architecture"] == name], float)
            means.append(values.mean())
            errors.append(stats.sem(values) if len(values) > 1 else 0.0)
        positions = np.arange(len(architectures))
        axis.bar(
            positions,
            means,
            yerr=errors,
            color=[colors[name] for name in architectures],
            capsize=3,
            alpha=0.88,
        )
        axis.set_xticks(positions, [labels[name] for name in architectures], rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fidelity (higher is better)")
    axes[1].set_ylabel("MAE (lower is better)")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def validate_completed(rows: list[dict], protocol: dict, preset: str) -> dict:
    expected_replicates = set(protocol["system"][f"{preset}_replicates"])
    expected = {
        (float(field), int(replicate), architecture)
        for field in protocol["system"]["fields_h_over_J"]
        for replicate in expected_replicates
        for architecture in protocol["architectures"]
    }
    observed = {(r["field"], r["replicate"], r["architecture"]) for r in rows}
    checks = {
        "all_expected_runs_present": observed == expected,
        "no_duplicate_runs": len(observed) == len(rows),
        "all_training_shots_equal": len({r["training_shots"] for r in rows}) == 1,
        "all_objective_queries_equal": len({r["objective_queries"] for r in rows}) == 1,
    }
    by_cell: dict[tuple[float, int], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["field"], row["replicate"]), []).append(row)
    checks["common_base_seed_within_cell"] = all(
        len({row["base_seed"] for row in cell}) == 1 for cell in by_cell.values()
    )
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "missing_runs": [list(x) for x in sorted(expected - observed, key=str)],
        "unexpected_runs": [list(x) for x in sorted(observed - expected, key=str)],
    }


def analyze_report(report_path: Path, protocol_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    rows = report["run_level_summary"]
    validation = validate_completed(rows, protocol, report["preset"])
    if not validation["all_checks_passed"]:
        raise RuntimeError(f"Completed-run validation failed: {validation}")

    contrasts = []
    for scope, key in (("primary", "primary_contrasts"), ("secondary", "secondary_contrasts")):
        for specification in protocol[key]:
            contrasts.append(
                make_contrast(
                    rows,
                    specification["name"],
                    specification["challenger"],
                    specification["baseline"],
                    scope,
                )
            )
    by_name = {entry["name"]: entry for entry in contrasts}
    parameter_metric = by_name["parameter_matched_mera_vs_mps"]["metrics"]["zz_long_range_mae"]
    gate_metric = by_name["gate_and_parameter_matched_mera_vs_local"]["metrics"]["zz_long_range_mae"]
    decisions = {
        "parameter_matched_long_range_support": (
            parameter_metric["mean_difference"] < 0 and parameter_metric["t95_interval"][1] < 0
        ),
        "gate_matched_geometry_support": (
            gate_metric["mean_difference"] < 0 and gate_metric["t95_interval"][1] < 0
        ),
    }
    decisions["both_primary_long_range_rules_pass"] = all(decisions.values())
    output = {
        "status": "complete",
        "source_report": str(report_path),
        "protocol_sha256": report["protocol_sha256"],
        "preset": report["preset"],
        "completed_run_validation": validation,
        "contrasts": contrasts,
        "preregistered_decisions": decisions,
        "interpretation_guardrail": (
            "Only MERA_PM225 versus MPS is a direct parameter-matched canonical-MPS contrast. "
            "MERA versus BRICKWORK390 is a gate-and-coordinate-matched geometry control."
        ),
    }
    destination = report_path.parent / "paired_contrasts.json"
    atomic_json(destination, output)
    plot_frontier(rows, report_path.parent / "resource_frontier_summary.png")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parent / "protocols" / "resource_frontier_v2.json",
    )
    args = parser.parse_args()
    output = analyze_report(args.report, args.protocol)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
