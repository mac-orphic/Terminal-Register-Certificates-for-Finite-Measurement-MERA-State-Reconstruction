"""Exact differentiable statevector backend for the noiseless diagnostic."""

from __future__ import annotations

import math
import time

import numpy as np

from .core import Architecture, evaluate, random_parameters, stable_seed


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - user-facing dependency error
        raise RuntimeError(
            "PyTorch is required for the noiseless multistart diagnostic. "
            "Install the bundle requirements first."
        ) from exc
    return torch


def _constants(torch, device):
    complex_dtype = torch.complex128
    i4 = torch.eye(4, dtype=complex_dtype, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=complex_dtype, device=device)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=complex_dtype, device=device)
    z = torch.tensor([[1, 0], [0, -1]], dtype=complex_dtype, device=device)
    return i4, x, y, z


def _rotation(torch, p):
    phi, theta, omega = p[0], p[1], p[2]
    zero = torch.zeros((), dtype=torch.complex128, device=p.device)
    c = torch.cos(theta / 2).to(torch.complex128)
    s = torch.sin(theta / 2).to(torch.complex128)
    ry = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rz1 = torch.stack([
        torch.stack([torch.exp(-0.5j * phi), zero]),
        torch.stack([zero, torch.exp(0.5j * phi)]),
    ])
    rz2 = torch.stack([
        torch.stack([torch.exp(-0.5j * omega), zero]),
        torch.stack([zero, torch.exp(0.5j * omega)]),
    ])
    return rz2 @ ry @ rz1


def _ising(torch, angle, pauli, i4):
    pp = torch.kron(pauli, pauli)
    return torch.cos(angle / 2) * i4 - 1j * torch.sin(angle / 2) * pp


def _su4(torch, p, constants):
    i4, x, y, z = constants
    pre = torch.kron(_rotation(torch, p[0:3]), _rotation(torch, p[3:6]))
    post = torch.kron(_rotation(torch, p[9:12]), _rotation(torch, p[12:15]))
    return post @ _ising(torch, p[8], z, i4) @ _ising(torch, p[7], y, i4) @ _ising(torch, p[6], x, i4) @ pre


def _apply(torch, state, unitary, wires: tuple[int, int], n: int):
    rest = [wire for wire in range(n) if wire not in wires]
    permutation = list(wires) + rest
    inverse = list(np.argsort(permutation))
    tensor = state.reshape([2] * n).permute(*permutation).reshape(4, -1)
    tensor = unitary @ tensor
    return tensor.reshape([2] * n).permute(*inverse).reshape(1 << n)


def analyze_state_torch(target, theta, spec: Architecture, n: int):
    torch = require_torch()
    constants = _constants(torch, theta.device)
    state = target
    grouped = theta.reshape(spec.independent_groups, 15)
    for wires, group in zip(spec.blocks, spec.parameter_groups):
        state = _apply(torch, state, _su4(torch, grouped[group], constants), wires, n)
    return state


def hamming_weights_torch(n: int, device):
    torch = require_torch()
    indices = np.arange(1 << n, dtype=np.uint64)
    weights = np.zeros(1 << n, dtype=np.float64)
    for wire in range(1, n):
        weights += ((indices >> np.uint64(n - 1 - wire)) & np.uint64(1)).astype(float)
    return torch.tensor(weights, dtype=torch.float64, device=device)


def exact_objective(target, theta, spec: Architecture, n: int, weights):
    state = analyze_state_torch(target, theta, spec, n)
    return ((torch_abs_squared(state)) * weights).sum()


def torch_abs_squared(value):
    return value.real.square() + value.imag.square()


def train_exact_adam(
    *,
    target_numpy: np.ndarray,
    ground_energy: float,
    n: int,
    field: float,
    spec: Architecture,
    restart: int,
    namespace: str,
    match_group: str,
    optimizer_config: dict,
    device_name: str,
) -> dict:
    torch = require_torch()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    torch.set_default_dtype(torch.float64)

    base_seed = stable_seed(namespace, n, field, restart, match_group)
    initialization_seed = stable_seed(base_seed, "initialization")
    initial = random_parameters(spec, initialization_seed)
    theta = torch.tensor(initial, dtype=torch.float64, device=device, requires_grad=True)
    target = torch.tensor(target_numpy, dtype=torch.complex128, device=device)
    weights = hamming_weights_torch(n, device)

    maximum_steps = int(optimizer_config["maximum_steps"])
    grad_tolerance = float(optimizer_config["gradient_norm_tolerance"])
    relative_tolerance = float(optimizer_config["relative_objective_change_tolerance"])
    plateau_required = int(optimizer_config["plateau_steps"])
    schedule = optimizer_config["learning_rate_schedule"]
    optimizer = torch.optim.Adam([theta], lr=float(schedule[0]["learning_rate"]))

    def learning_rate(step: int) -> float:
        for segment in schedule:
            if int(segment["start_step"]) <= step <= int(segment["end_step"]):
                return float(segment["learning_rate"])
        raise RuntimeError(f"No learning rate defined for step {step}.")

    history = []
    previous = None
    plateau = 0
    converged = False
    started = time.perf_counter()
    final_grad_norm = math.inf
    completed_steps = 0

    for step in range(maximum_steps):
        lr = learning_rate(step)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = exact_objective(target, theta, spec, n, weights)
        loss.backward()
        final_grad_norm = float(torch.linalg.vector_norm(theta.grad).detach().cpu())
        value = float(loss.detach().cpu())

        if previous is not None:
            relative = abs(previous - value) / max(1.0, abs(previous))
            plateau = plateau + 1 if relative < relative_tolerance else 0
        previous = value

        if step % 100 == 0 or step + 1 == maximum_steps:
            history.append({
                "step": step,
                "exact_eta_sum": value,
                "gradient_norm": final_grad_norm,
                "learning_rate": lr,
                "plateau_counter": plateau,
            })

        completed_steps = step + 1
        if final_grad_norm < grad_tolerance and plateau >= plateau_required:
            converged = True
            break
        optimizer.step()

    # Re-evaluate at the returned parameters.  When the maximum-step branch is
    # taken, the last Adam update occurs after the last in-loop diagnostic.
    optimizer.zero_grad(set_to_none=True)
    final_loss = exact_objective(target, theta, spec, n, weights)
    final_loss.backward()
    final_grad_norm = float(torch.linalg.vector_norm(theta.grad).detach().cpu())
    final_objective = float(final_loss.detach().cpu())
    history.append({
        "step": completed_steps,
        "exact_eta_sum": final_objective,
        "gradient_norm": final_grad_norm,
        "learning_rate": learning_rate(min(completed_steps - 1, maximum_steps - 1)),
        "plateau_counter": plateau,
        "returned_parameter_diagnostic": True,
    })
    final_theta = theta.detach().cpu().numpy()
    metrics = evaluate(target_numpy, ground_energy, final_theta, spec, n, field)
    return {
        "status": "complete",
        "restart": restart,
        "device": str(device),
        "seeds": {"base": base_seed, "initialization": initialization_seed},
        "converged": converged,
        "completed_steps": completed_steps,
        "final_gradient_norm": final_grad_norm,
        "final_exact_eta_sum": final_objective,
        "plateau_counter": plateau,
        "history": history,
        "evaluation": metrics,
        "trained_parameters": final_theta.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
