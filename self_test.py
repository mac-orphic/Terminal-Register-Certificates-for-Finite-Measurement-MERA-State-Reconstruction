#!/usr/bin/env python3
"""Fast structural and numerical checks for resource-frontier v2."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from qipmera.core import (
    analyze_batch,
    architecture,
    conditional_top,
    exact_costs,
    generate,
    hamming_weights,
    random_parameters,
    stable_seed,
    tfim_ground_state,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    protocol = json.loads((ROOT / "protocols" / "resource_frontier_v2.json").read_text())
    checks: dict[str, bool] = {}
    n = int(protocol["system"]["n"])

    for name, expected in protocol["resource_table"].items():
        spec = architecture(n, name)
        checks[f"{name}_block_count"] = spec.physical_blocks == expected["physical_su4_blocks"]
        checks[f"{name}_coordinate_count"] = spec.coordinates == expected["independent_coordinates"]
        checks[f"{name}_valid_wires"] = all(
            0 <= a < n and 0 <= b < n and a != b for a, b in spec.blocks
        )
        if name in ("MPS", "BRICKWORK225", "BRICKWORK390"):
            checks[f"{name}_nearest_neighbor"] = all(abs(a - b) == 1 for a, b in spec.blocks)

    checks["parameter_match_MERA_PM225_MPS"] = (
        architecture(n, "MERA_PM225").coordinates == architecture(n, "MPS").coordinates == 225
    )
    checks["gate_coordinate_match_MERA_BRICKWORK390"] = (
        architecture(n, "MERA").physical_blocks
        == architecture(n, "BRICKWORK390").physical_blocks
        == 26
        and architecture(n, "MERA").coordinates
        == architecture(n, "BRICKWORK390").coordinates
        == 390
    )
    pilot = set(protocol["system"]["pilot_replicates"])
    full = set(protocol["system"]["full_replicates"])
    checks["pilot_full_replicates_disjoint"] = pilot.isdisjoint(full)

    stages = protocol["training_stages"]
    queries = sum(2 * int(s["directions"]) * int(s["steps"]) for s in stages)
    shots = sum(
        2 * int(s["directions"]) * int(s["steps"]) * int(s["shots"])
        for s in stages
    )
    checks["query_budget"] = queries == 40_000
    checks["shot_budget"] = shots == 560_000_000

    # Full round trips for the two newly decisive matched models.  The test is
    # exact and verifies that the terminal all-zero event and reconstruction
    # logic remain correct despite sharing/repeated local blocks.
    for name in ("MERA_PM225", "BRICKWORK390"):
        spec = architecture(n, name)
        theta = random_parameters(spec, stable_seed("frontier_v2_roundtrip", name))
        top = np.asarray([0.6, 0.8j], dtype=complex)
        top /= np.linalg.norm(top)
        state = generate(top, theta, spec, n)
        analyzed = analyze_batch(state, theta, spec, n)[0]
        recovered_top, p0 = conditional_top(analyzed, n)
        reconstructed = generate(recovered_top, theta, spec, n)
        checks[f"{name}_roundtrip"] = (
            abs(p0 - 1.0) < 1e-10
            and abs(abs(np.vdot(state, reconstructed)) ** 2 - 1.0) < 1e-10
        )
        eta_sum = float(exact_costs(analyzed[None, :], hamming_weights(n))[0])
        checks[f"{name}_certificate_zero"] = abs(eta_sum) < 1e-10

    with tempfile.TemporaryDirectory() as directory:
        target, _ = tfim_ground_state(4, 1.0, Path(directory))
        checks["tfim_target_normalized"] = abs(float(np.vdot(target, target).real) - 1.0) < 1e-10

    checks = {key: bool(value) for key, value in checks.items()}
    result = {"all_checks_passed": all(checks.values()), "checks": checks}
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
