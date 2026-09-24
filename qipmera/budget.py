"""Checkpointed finite-shot training and terminal-certificate utilities."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .core import (
    Architecture,
    _objective_values,
    analyze_batch,
    architecture,
    evaluate,
    hamming_weights,
    random_parameters,
    stable_seed,
    tfim_ground_state,
)


def checkpointed_spsa_train(
    target: np.ndarray,
    ground_energy: float,
    n: int,
    field: float,
    spec: Architecture,
    initial: np.ndarray,
    stages: list[dict],
    requested_budgets: list[int],
    measurement_seed: int,
    direction_seed: int,
) -> tuple[list[dict], dict]:
    """Run one frozen trajectory and evaluate at cumulative-shot checkpoints.

    A checkpoint is taken after the first completed SPSA step that reaches or
    exceeds its requested budget.  This preserves the stage schedule exactly;
    the achieved budget is retained whenever the discrete step cost prevents
    equality.
    """
    requested = sorted({int(value) for value in requested_budgets})
    if not requested or requested[0] <= 0:
        raise ValueError("All requested budgets must be positive.")

    maximum_available = sum(
        2 * int(stage["directions"]) * int(stage["steps"]) * int(stage["shots"])
        for stage in stages
    )
    if requested[-1] > maximum_available:
        raise ValueError(
            f"Largest checkpoint {requested[-1]} exceeds trajectory budget {maximum_available}."
        )

    theta = initial.copy()
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    weights = hamming_weights(n)
    global_step = 0
    total_queries = 0
    total_shots = 0
    next_checkpoint = 0
    checkpoints: list[dict] = []
    started = time.perf_counter()

    for stage_index, stage in enumerate(stages):
        measurement_rng = np.random.default_rng(stable_seed(measurement_seed, stage_index))
        direction_rng = np.random.default_rng(stable_seed(direction_seed, stage_index))
        steps = int(stage["steps"])
        directions = int(stage["directions"])
        shots = int(stage["shots"])

        for step in range(steps):
            ck = float(stage["perturbation"]) / ((step + 1) ** 0.101)
            deltas = direction_rng.choice(
                (-1.0, 1.0), size=(directions, len(theta))
            )
            candidates = np.empty((2 * directions, len(theta)))
            candidates[0::2] = theta + ck * deltas
            candidates[1::2] = theta - ck * deltas
            values = _objective_values(
                target, candidates, spec, n, weights, shots, measurement_rng
            )
            gradient = (
                ((values[0::2] - values[1::2]) / (2 * ck))[:, None] * deltas
            ).mean(axis=0)

            total_queries += 2 * directions
            total_shots += 2 * directions * shots
            global_step += 1

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            first_hat = first / (1 - 0.9**global_step)
            second_hat = second / (1 - 0.999**global_step)
            rate = float(stage["learning_rate"]) / ((step + 1) ** 0.15)
            theta -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)

            while (
                next_checkpoint < len(requested)
                and total_shots >= requested[next_checkpoint]
            ):
                metrics = evaluate(target, ground_energy, theta, spec, n, field)
                checkpoints.append(
                    {
                        "requested_training_shots": requested[next_checkpoint],
                        "achieved_training_shots": total_shots,
                        "objective_queries": total_queries,
                        "stage": stage_index,
                        "stage_step_completed": step + 1,
                        "global_step_completed": global_step,
                        "elapsed_seconds": time.perf_counter() - started,
                        "evaluation": metrics,
                        "trained_parameters": theta.tolist(),
                    }
                )
                next_checkpoint += 1

            if next_checkpoint == len(requested):
                break
        if next_checkpoint == len(requested):
            break

    if next_checkpoint != len(requested):
        raise RuntimeError("Training ended before all requested checkpoints were reached.")

    resources = {
        "objective_queries": total_queries,
        "training_shots": total_shots,
        "maximum_available_training_shots": maximum_available,
        "exact_objective": False,
    }
    return checkpoints, resources


def run_budget_case(
    *,
    n: int,
    field: float,
    replicate: int,
    architecture_name: str,
    stages: list[dict],
    requested_budgets: list[int],
    cache_dir: Path,
    seed_namespace: str,
    seed_match_group: str,
) -> dict:
    target, energy = tfim_ground_state(n, field, cache_dir)
    spec = architecture(n, architecture_name)
    base = stable_seed(seed_namespace, n, field, replicate, seed_match_group)
    initialization_seed = stable_seed(base, "initialization")
    measurement_seed = stable_seed(base, "measurement")
    direction_seed = stable_seed(base, "direction")
    initial = random_parameters(spec, initialization_seed)
    checkpoints, resources = checkpointed_spsa_train(
        target,
        energy,
        n,
        field,
        spec,
        initial,
        stages,
        requested_budgets,
        measurement_seed,
        direction_seed,
    )
    return {
        "status": "complete",
        "case": {
            "n": n,
            "field_h_over_J": field,
            "replicate": replicate,
            "architecture": architecture_name,
            "optimizer": "global_spsa_adam_moments",
        },
        "seeds": {
            "base": base,
            "initialization": initialization_seed,
            "measurement": measurement_seed,
            "direction": direction_seed,
        },
        "architecture_resources": {
            "physical_su4_blocks": spec.physical_blocks,
            "independent_parameter_groups": spec.independent_groups,
            "independent_coordinates": spec.coordinates,
        },
        "training_resources_at_last_checkpoint": resources,
        "checkpoints": checkpoints,
    }


def terminal_certificate_samples(
    *,
    p0: float,
    infidelity: float,
    certificate_shots: int,
    delta: float,
    tau: float,
    epsilon_top: float,
    repetitions: int,
    seed: int,
) -> list[dict]:
    """Draw repeated terminal all-zero counts and form one-sided bounds."""
    if not 0 <= p0 <= 1:
        raise ValueError("p0 must lie in [0,1].")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1).")
    if certificate_shots <= 0 or repetitions <= 0:
        raise ValueError("Shots and repetitions must be positive.")
    rng = np.random.default_rng(seed)
    successes = rng.binomial(certificate_shots, p0, size=repetitions)
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * certificate_shots))
    output = []
    for repetition, count in enumerate(successes):
        p0_hat = float(count / certificate_shots)
        upper = min(1.0, 1.0 - p0_hat + radius + tau + epsilon_top)
        output.append(
            {
                "certificate_repetition": repetition,
                "certificate_shots": certificate_shots,
                "all_zero_count": int(count),
                "p0_hat": p0_hat,
                "hoeffding_radius": radius,
                "certificate_upper_bound": upper,
                "covers_exact_infidelity": bool(upper + 1e-15 >= infidelity),
            }
        )
    return output
