"""Numerical core shared by every experiment in the bundle.

The implementation uses dense statevectors and batched two-qubit gates.  It is
intended for n <= 16 on an ordinary workstation.  All reported reconstruction
metrics are evaluated exactly; only the training objective is sampled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

I4 = np.eye(4, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def stable_seed(*labels: object) -> int:
    raw = "|".join(map(str, labels)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


@dataclass(frozen=True)
class Architecture:
    name: str
    blocks: tuple[tuple[int, int], ...]
    parameter_groups: tuple[int, ...]
    discarded: tuple[int, ...]

    @property
    def physical_blocks(self) -> int:
        return len(self.blocks)

    @property
    def independent_groups(self) -> int:
        return max(self.parameter_groups) + 1

    @property
    def coordinates(self) -> int:
        return 15 * self.independent_groups


def _binary_layers(n: int) -> list[dict]:
    if n < 2 or n & (n - 1):
        raise ValueError("Binary MERA/TTN requires n to be a power of two.")
    active = list(range(n))
    layers: list[dict] = []
    while len(active) > 1:
        disentanglers = [(active[k], active[k + 1]) for k in range(1, len(active) - 1, 2)]
        isometries = [(active[k], active[k + 1]) for k in range(0, len(active), 2)]
        layers.append({"disentanglers": disentanglers, "isometries": isometries})
        active = [left for left, _ in isometries]
    return layers


def architecture(n: int, name: str) -> Architecture:
    name = name.upper()
    discarded = tuple(range(n - 1, 0, -1))
    if name == "MPS":
        blocks = tuple((i - 1, i) for i in range(n - 1, 0, -1))
        return Architecture(name, blocks, tuple(range(len(blocks))), discarded)
    if name == "BRICKWORK225":
        if n != 16:
            raise ValueError("BRICKWORK225 is frozen for n=16.")
        pool: list[tuple[int, int]] = []
        layer = 0
        while len(pool) < 15:
            start = layer % 2
            pool.extend((i, i + 1) for i in range(start, n - 1, 2))
            layer += 1
        blocks = tuple(pool[:15])
        return Architecture(name, blocks, tuple(range(15)), discarded)
    if name == "BRICKWORK390":
        if n != 16:
            raise ValueError("BRICKWORK390 is frozen for n=16.")
        # Gate- and coordinate-matched local control for the full MERA:
        # 26 nearest-neighbour SU(4) blocks and 26 independent parameter
        # groups.  This is deliberately called a brickwork control, not an
        # MPS, because the added trainable layers change the canonical chi=2
        # MPS family.
        pool: list[tuple[int, int]] = []
        layer = 0
        while len(pool) < 26:
            start = layer % 2
            bonds = [(i, i + 1) for i in range(start, n - 1, 2)]
            if (layer // 2) % 2:
                bonds.reverse()
            pool.extend(bonds)
            layer += 1
        blocks = tuple(pool[:26])
        return Architecture(name, blocks, tuple(range(26)), discarded)
    if name == "LOCAL_LC225":
        if n != 16:
            raise ValueError("LOCAL_LC225 is frozen for n=16.")
        # Nine complete alternating nearest-neighbour layers are the minimum
        # needed for the backward causal cones of every pair at distance
        # r >= 8 to overlap on an open 16-site chain.  Gates acting on the
        # same bond share one SU(4) parameter group, giving 15 groups = 225
        # independent coordinates while retaining all 68 physical blocks.
        blocks: list[tuple[int, int]] = []
        groups: list[int] = []
        for layer in range(9):
            start = layer % 2
            for left in range(start, n - 1, 2):
                blocks.append((left, left + 1))
                groups.append(left)
        return Architecture(name, tuple(blocks), tuple(groups), discarded)

    layers = _binary_layers(n)
    if name == "TTN":
        blocks = tuple(block for layer in layers for block in layer["isometries"])
        return Architecture(name, blocks, tuple(range(len(blocks))), discarded)
    if name == "MERA":
        blocks = tuple(block for layer in layers for role in ("disentanglers", "isometries") for block in layer[role])
        return Architecture(name, blocks, tuple(range(len(blocks))), discarded)
    if name == "MERA_PM225":
        if n != 16:
            raise ValueError("MERA_PM225 is frozen for n=16.")
        blocks: list[tuple[int, int]] = []
        groups: list[int] = []
        # Exactly 15 independent SU(4) parameter groups = 225 coordinates.
        # Disentanglers share one group per nontrivial layer (groups 12--14).
        # Isometries use 8 + 2 + 1 + 1 groups (groups 0--11).
        iso_group_patterns = [list(range(8)), [8, 9, 8, 9], [10, 10], [11]]
        for ell, layer in enumerate(layers):
            for block in layer["disentanglers"]:
                blocks.append(block); groups.append(12 + ell)
            pattern = iso_group_patterns[ell]
            for j, block in enumerate(layer["isometries"]):
                blocks.append(block); groups.append(pattern[j])
        return Architecture(name, tuple(blocks), tuple(groups), discarded)
    raise ValueError(f"Unknown architecture: {name}")


def rot_batch(p: np.ndarray) -> np.ndarray:
    phi, theta, omega = p[:, 0], p[:, 1], p[:, 2]
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry = np.zeros((len(p), 2, 2), complex)
    ry[:, 0, 0] = c; ry[:, 0, 1] = -s
    ry[:, 1, 0] = s; ry[:, 1, 1] = c
    rz1 = np.zeros_like(ry); rz2 = np.zeros_like(ry)
    rz1[:, 0, 0] = np.exp(-0.5j * phi); rz1[:, 1, 1] = np.exp(0.5j * phi)
    rz2[:, 0, 0] = np.exp(-0.5j * omega); rz2[:, 1, 1] = np.exp(0.5j * omega)
    return rz2 @ ry @ rz1


def kron_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bkl->bikjl", a, b).reshape(len(a), 4, 4)


def ising_batch(angle: np.ndarray, pauli: np.ndarray) -> np.ndarray:
    pp = np.kron(pauli, pauli)
    return np.cos(angle / 2)[:, None, None] * I4 - 1j * np.sin(angle / 2)[:, None, None] * pp


def su4_batch(p: np.ndarray) -> np.ndarray:
    pre = kron_batch(rot_batch(p[:, 0:3]), rot_batch(p[:, 3:6]))
    post = kron_batch(rot_batch(p[:, 9:12]), rot_batch(p[:, 12:15]))
    return post @ ising_batch(p[:, 8], Z) @ ising_batch(p[:, 7], Y) @ ising_batch(p[:, 6], X) @ pre


def apply_batch(states: np.ndarray, unitaries: np.ndarray, wires: tuple[int, int], n: int) -> np.ndarray:
    batch = len(states)
    rest = tuple(w for w in range(n) if w not in wires)
    permutation = (0,) + tuple(w + 1 for w in wires) + tuple(w + 1 for w in rest)
    inverse = np.argsort(permutation)
    tensor = states.reshape((batch,) + (2,) * n).transpose(permutation).reshape(batch, 4, -1)
    tensor = np.einsum("bij,bjk->bik", unitaries, tensor)
    return tensor.reshape((batch,) + (2,) * n).transpose(inverse).reshape(batch, 1 << n)


def analyze_batch(target: np.ndarray, parameters: np.ndarray, spec: Architecture, n: int) -> np.ndarray:
    parameters = np.atleast_2d(parameters)
    if parameters.shape[1] != spec.coordinates:
        raise ValueError(f"Expected {spec.coordinates} parameters, got {parameters.shape[1]}.")
    states = np.repeat(target[None, :], len(parameters), axis=0)
    grouped = parameters.reshape(len(parameters), spec.independent_groups, 15)
    for wires, group in zip(spec.blocks, spec.parameter_groups):
        states = apply_batch(states, su4_batch(grouped[:, group]), wires, n)
    return states


def generate(top: np.ndarray, parameters: np.ndarray, spec: Architecture, n: int) -> np.ndarray:
    state = np.zeros((1, 1 << n), complex)
    state[0, 0] = top[0]
    state[0, 1 << (n - 1)] = top[1]
    grouped = np.asarray(parameters).reshape(spec.independent_groups, 15)
    for j in reversed(range(spec.physical_blocks)):
        gate = su4_batch(grouped[spec.parameter_groups[j]:spec.parameter_groups[j] + 1]).conj().transpose(0, 2, 1)
        state = apply_batch(state, gate, spec.blocks[j], n)
    return state[0]


def hamming_weights(n: int) -> np.ndarray:
    indices = np.arange(1 << n, dtype=np.uint64)
    result = np.zeros(1 << n, dtype=np.int16)
    for wire in range(1, n):
        result += ((indices >> np.uint64(n - 1 - wire)) & np.uint64(1)).astype(np.int16)
    return result


def exact_costs(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.abs(states) ** 2 @ weights


def sampled_costs(states: np.ndarray, weights: np.ndarray, shots: int | None, rng: np.random.Generator) -> np.ndarray:
    if shots is None:
        return exact_costs(states, weights)
    values = []
    for probabilities in np.abs(states) ** 2:
        mass = np.bincount(weights, weights=probabilities, minlength=int(weights.max()) + 1)
        mass = np.clip(mass, 0, None); mass /= mass.sum()
        counts = rng.multinomial(int(shots), mass)
        values.append(float(counts @ np.arange(len(mass))) / shots)
    return np.asarray(values)


def p0_value(state: np.ndarray, n: int) -> float:
    return float(abs(state[0]) ** 2 + abs(state[1 << (n - 1)]) ** 2)


def conditional_top(state: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    top = np.array([state[0], state[1 << (n - 1)]], complex)
    probability = float(np.vdot(top, top).real)
    if probability <= 1e-15:
        raise RuntimeError("Terminal all-zero event has zero probability.")
    return top / np.sqrt(probability), probability


def random_parameters(spec: Architecture, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.30, size=spec.coordinates)


def tfim_ground_state(n: int, field: float, cache_dir: Path | None = None) -> tuple[np.ndarray, float]:
    if n > 20:
        raise MemoryError("Dense backend is intentionally limited to n<=20. Use a tensor-network backend above this size.")
    cache_path = None if cache_dir is None else cache_dir / f"tfim_n{n:02d}_h{field:.6f}.npz"
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return data["state"], float(data["energy"])
    dimension = 1 << n
    indices = np.arange(dimension, dtype=np.uint64)
    diagonal = np.zeros(dimension, float)
    for i in range(n - 1):
        a = (indices >> np.uint64(n - 1 - i)) & np.uint64(1)
        b = (indices >> np.uint64(n - 2 - i)) & np.uint64(1)
        diagonal -= np.where(a == b, 1.0, -1.0)
    flips = [indices ^ np.uint64(1 << (n - 1 - wire)) for wire in range(n)]

    def matvec(vector: np.ndarray) -> np.ndarray:
        out = diagonal * vector
        for flipped in flips:
            out -= field * vector[flipped]
        return out

    operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=float)
    v0 = np.random.default_rng(stable_seed("ground", n, field)).normal(size=dimension)
    values, vectors = eigsh(operator, k=1, which="SA", v0=v0, tol=1e-11, maxiter=10000)
    state = vectors[:, 0].astype(complex)
    pivot = int(np.argmax(abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    if state[pivot].real < 0:
        state = -state
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, state=state, energy=np.asarray(values[0]))
        os.replace(temporary, cache_path)
    return state, float(values[0])


def _objective_values(target: np.ndarray, candidates: np.ndarray, spec: Architecture, n: int,
                      weights: np.ndarray, shots: int | None, rng: np.random.Generator) -> np.ndarray:
    return sampled_costs(analyze_batch(target, candidates, spec, n), weights, shots, rng)


def spsa_train(target: np.ndarray, n: int, spec: Architecture, initial: np.ndarray,
               stages: list[dict], measurement_seed: int, direction_seed: int,
               blockwise: bool = False) -> tuple[np.ndarray, list[dict], dict]:
    theta = initial.copy()
    first = np.zeros_like(theta); second = np.zeros_like(theta)
    weights = hamming_weights(n); global_step = 0; history: list[dict] = []
    total_queries = 0; total_shots = 0
    for stage_index, stage in enumerate(stages):
        measurement_rng = np.random.default_rng(stable_seed(measurement_seed, stage_index))
        direction_rng = np.random.default_rng(stable_seed(direction_seed, stage_index))
        steps = int(stage["steps"]); directions = int(stage["directions"])
        shots = stage.get("shots")
        shots = None if shots in (None, "exact") else int(shots)
        for step in range(steps):
            if step % max(1, steps // 10) == 0:
                analyzed = analyze_batch(target, theta, spec, n)[0]
                history.append({"stage": stage_index, "step": step, "global_step": global_step,
                                "exact_eta_sum": float(exact_costs(analyzed[None, :], weights)[0]),
                                "joint_failure": 1.0 - p0_value(analyzed, n)})
            ck = float(stage["perturbation"]) / ((step + 1) ** 0.101)
            deltas = np.zeros((directions, len(theta)))
            if blockwise:
                group = global_step % spec.independent_groups
                sl = slice(15 * group, 15 * (group + 1))
                deltas[:, sl] = direction_rng.choice((-1.0, 1.0), size=(directions, 15))
            else:
                deltas[:] = direction_rng.choice((-1.0, 1.0), size=deltas.shape)
            batch = np.empty((2 * directions, len(theta)))
            batch[0::2] = theta + ck * deltas
            batch[1::2] = theta - ck * deltas
            values = _objective_values(target, batch, spec, n, weights, shots, measurement_rng)
            gradient = (((values[0::2] - values[1::2]) / (2 * ck))[:, None] * deltas).mean(axis=0)
            total_queries += 2 * directions
            if shots is not None:
                total_shots += 2 * directions * shots
            global_step += 1
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient ** 2
            first_hat = first / (1 - 0.9 ** global_step)
            second_hat = second / (1 - 0.999 ** global_step)
            rate = float(stage["learning_rate"]) / ((step + 1) ** 0.15)
            theta -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        analyzed = analyze_batch(target, theta, spec, n)[0]
        history.append({"stage": stage_index, "step": steps, "global_step": global_step,
                        "exact_eta_sum": float(exact_costs(analyzed[None, :], weights)[0]),
                        "joint_failure": 1.0 - p0_value(analyzed, n)})
    return theta, history, {"objective_queries": total_queries, "training_shots": total_shots,
                            "exact_objective": all(s.get("shots") in (None, "exact") for s in stages)}


def parameter_shift_train(target: np.ndarray, n: int, spec: Architecture, initial: np.ndarray,
                          steps: int, shots: int, learning_rate: float,
                          measurement_seed: int) -> tuple[np.ndarray, list[dict], dict]:
    """Budget-transparent full parameter-shift Adam control.

    Each coordinate occurs once in its independent SU(4) group.  The elementary
    rotations use generators with eigenvalues +/-1/2, hence a pi/2 shift.
    """
    theta = initial.copy(); first = np.zeros_like(theta); second = np.zeros_like(theta)
    weights = hamming_weights(n); rng = np.random.default_rng(measurement_seed)
    history: list[dict] = []; shift = np.pi / 2
    eye = np.eye(len(theta)) * shift
    for step in range(steps):
        candidates = np.concatenate([theta + eye, theta - eye], axis=0)
        values = _objective_values(target, candidates, spec, n, weights, shots, rng)
        gradient = 0.5 * (values[:len(theta)] - values[len(theta):])
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        first_hat = first / (1 - 0.9 ** (step + 1)); second_hat = second / (1 - 0.999 ** (step + 1))
        theta -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        if step % max(1, steps // 10) == 0 or step + 1 == steps:
            analyzed = analyze_batch(target, theta, spec, n)[0]
            history.append({"step": step + 1, "exact_eta_sum": float(exact_costs(analyzed[None], weights)[0]),
                            "joint_failure": 1.0 - p0_value(analyzed, n)})
    queries = 2 * len(theta) * steps
    return theta, history, {"objective_queries": queries, "training_shots": queries * shots,
                            "exact_objective": False}


def expectation_z(state: np.ndarray, n: int, wire: int) -> float:
    probabilities = np.abs(state) ** 2
    indices = np.arange(1 << n, dtype=np.uint64)
    bits = (indices >> np.uint64(n - 1 - wire)) & np.uint64(1)
    return float(probabilities @ np.where(bits == 0, 1.0, -1.0))


def expectation_zz(state: np.ndarray, n: int, i: int, j: int) -> float:
    probabilities = np.abs(state) ** 2
    indices = np.arange(1 << n, dtype=np.uint64)
    a = (indices >> np.uint64(n - 1 - i)) & np.uint64(1)
    b = (indices >> np.uint64(n - 1 - j)) & np.uint64(1)
    return float(probabilities @ np.where(a == b, 1.0, -1.0))


def expectation_x(state: np.ndarray, n: int, wire: int) -> float:
    indices = np.arange(1 << n, dtype=np.uint64)
    flipped = indices ^ np.uint64(1 << (n - 1 - wire))
    return float(np.vdot(state, state[flipped]).real)


def expectation_xx(state: np.ndarray, n: int, i: int, j: int) -> float:
    indices = np.arange(1 << n, dtype=np.uint64)
    mask = np.uint64((1 << (n - 1 - i)) | (1 << (n - 1 - j)))
    return float(np.vdot(state, state[indices ^ mask]).real)


def correlation_profiles(state: np.ndarray, n: int) -> dict[str, list[float]]:
    z = [expectation_z(state, n, i) for i in range(n)]
    x = [expectation_x(state, n, i) for i in range(n)]
    zz_raw: list[float] = []; zz_connected: list[float] = []; xx_connected: list[float] = []
    for distance in range(1, n):
        zzr = []; zzc = []; xxc = []
        for i in range(n - distance):
            j = i + distance
            zz_value = expectation_zz(state, n, i, j)
            xx_value = expectation_xx(state, n, i, j)
            zzr.append(zz_value); zzc.append(zz_value - z[i] * z[j]); xxc.append(xx_value - x[i] * x[j])
        zz_raw.append(float(np.mean(zzr))); zz_connected.append(float(np.mean(zzc))); xx_connected.append(float(np.mean(xxc)))
    return {"zz_raw": zz_raw, "zz_connected": zz_connected, "xx_connected": xx_connected}


def energy_density(state: np.ndarray, n: int, field: float) -> float:
    return (-sum(expectation_zz(state, n, i, i + 1) for i in range(n - 1))
            - field * sum(expectation_x(state, n, i) for i in range(n))) / n


def half_chain_entropy(state: np.ndarray, n: int) -> float:
    matrix = state.reshape(1 << (n // 2), 1 << (n - n // 2))
    singular = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular ** 2
    probabilities = probabilities[probabilities > 1e-15]
    return float(-(probabilities * np.log2(probabilities)).sum())


def effective_correlation_length(profile: Iterable[float], n: int) -> dict:
    values = np.abs(np.asarray(list(profile), float))
    distances = np.arange(1, len(values) + 1)
    mask = (distances <= n // 2) & (values > 1e-10)
    if mask.sum() < 3:
        return {"xi": None, "r2": None, "status": "insufficient_points"}
    slope, intercept = np.polyfit(distances[mask], np.log(values[mask]), 1)
    prediction = slope * distances[mask] + intercept
    residual = float(np.sum((np.log(values[mask]) - prediction) ** 2))
    total = float(np.sum((np.log(values[mask]) - np.log(values[mask]).mean()) ** 2))
    xi = None if slope >= 0 else float(-1 / slope)
    return {"xi": xi, "r2": None if total == 0 else float(1 - residual / total),
            "status": "ok" if xi is not None else "nondecaying_fit"}


def evaluate(target: np.ndarray, ground_energy: float, theta: np.ndarray,
             spec: Architecture, n: int, field: float) -> dict:
    analyzed = analyze_batch(target, theta, spec, n)[0]
    top, p0 = conditional_top(analyzed, n)
    reconstruction = generate(top, theta, spec, n)
    target_profiles = correlation_profiles(target, n)
    reconstructed_profiles = correlation_profiles(reconstruction, n)
    distances = np.arange(1, n)
    long_mask = distances >= math.ceil(n / 2)
    errors = {}
    for key in target_profiles:
        difference = np.abs(np.asarray(target_profiles[key]) - np.asarray(reconstructed_profiles[key]))
        errors[key] = {"all_distance_mae": float(difference.mean()),
                       "long_range_mae": float(difference[long_mask].mean())}
    target_entropy = half_chain_entropy(target, n); reconstructed_entropy = half_chain_entropy(reconstruction, n)
    target_mx = float(np.mean([expectation_x(target, n, i) for i in range(n)]))
    reconstructed_mx = float(np.mean([expectation_x(reconstruction, n, i) for i in range(n)]))
    target_energy_density = ground_energy / n
    reconstructed_energy_density = energy_density(reconstruction, n, field)
    weights = hamming_weights(n)
    return {
        "fidelity": float(abs(np.vdot(target, reconstruction)) ** 2),
        "p0": p0,
        "joint_failure": 1.0 - p0,
        "eta_sum": float(exact_costs(analyzed[None], weights)[0]),
        "certificate_slack_over_infidelity": float(exact_costs(analyzed[None], weights)[0]) - (1 - float(abs(np.vdot(target, reconstruction)) ** 2)),
        "energy_density_error": abs(reconstructed_energy_density - target_energy_density),
        "transverse_magnetization_error": abs(reconstructed_mx - target_mx),
        "half_chain_entropy_error": abs(reconstructed_entropy - target_entropy),
        "target_half_chain_entropy": target_entropy,
        "reconstructed_half_chain_entropy": reconstructed_entropy,
        "correlation_errors": errors,
        "target_effective_xi_zz": effective_correlation_length(target_profiles["zz_connected"], n),
        "reconstructed_effective_xi_zz": effective_correlation_length(reconstructed_profiles["zz_connected"], n),
        "held_out_definition": "distances r >= ceil(n/2); exact oracle evaluation after finite-shot training",
    }


def run_training_case(*, n: int, field: float, replicate: int, architecture_name: str,
                      stages: list[dict], output_path: Path, protocol_hash: str,
                      optimizer: str = "global_spsa", parameter_shift: dict | None = None,
                      cache_dir: Path | None = None, seed_namespace: str = "qip_followup_v1",
                      seed_match_group: str | None = None) -> dict:
    if output_path.exists():
        old = json.loads(output_path.read_text())
        if old.get("protocol_sha256") == protocol_hash and old.get("status") == "complete":
            return old
        raise RuntimeError(f"Existing checkpoint does not match protocol: {output_path}")
    target, energy = tfim_ground_state(n, field, cache_dir)
    spec = architecture(n, architecture_name)
    # Deliberately exclude optimizer/configuration from the base seed.  Within
    # an architecture-field-replicate cell, shot levels and optimizer methods
    # therefore start from exactly the same parameters.
    # Resource-frontier comparisons may request common random numbers across
    # architectures.  Architectures with equal coordinate counts then receive
    # exactly the same initial vector, SPSA directions, and sampling stream.
    # For unequal coordinate counts, the shorter vector is the common prefix.
    seed_label = architecture_name if seed_match_group is None else seed_match_group
    base = stable_seed(seed_namespace, n, field, replicate, seed_label)
    initial = random_parameters(spec, stable_seed(base, "initialization"))
    start = time.perf_counter()
    if optimizer == "parameter_shift":
        if parameter_shift is None:
            raise ValueError("parameter_shift configuration is required")
        theta, history, resources = parameter_shift_train(
            target, n, spec, initial, int(parameter_shift["steps"]), int(parameter_shift["shots"]),
            float(parameter_shift["learning_rate"]), stable_seed(base, "measurement"))
    else:
        theta, history, resources = spsa_train(
            target, n, spec, initial, stages, stable_seed(base, "measurement"),
            stable_seed(base, "direction"), blockwise=optimizer == "block_spsa")
    result = {
        "status": "complete", "protocol_sha256": protocol_hash,
        "case": {"n": n, "field_h_over_J": field, "replicate": replicate,
                 "architecture": architecture_name, "optimizer": optimizer},
        "seeds": {"base": base, "initialization": stable_seed(base, "initialization"),
                  "measurement": stable_seed(base, "measurement"), "direction": stable_seed(base, "direction")},
        "seed_match_group": seed_match_group,
        "architecture_resources": {"physical_su4_blocks": spec.physical_blocks,
                                   "independent_parameter_groups": spec.independent_groups,
                                   "independent_coordinates": spec.coordinates},
        "training_resources": resources,
        "stages": stages,
        "history": history,
        "evaluation": evaluate(target, energy, theta, spec, n, field),
        "trained_parameters": theta.tolist(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    atomic_json(output_path, result)
    return result
