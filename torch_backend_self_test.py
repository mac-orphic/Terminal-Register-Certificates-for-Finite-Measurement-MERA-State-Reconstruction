#!/usr/bin/env python3
"""Verify the exact PyTorch circuit and shared-parameter gradients."""

from __future__ import annotations

import json

import numpy as np

from qipmera.core import Architecture, analyze_batch, exact_costs, hamming_weights, random_parameters
from qipmera.exact_torch import analyze_state_torch, exact_objective, hamming_weights_torch, require_torch


def main():
    torch = require_torch(); torch.set_default_dtype(torch.float64)
    n = 3
    spec = Architecture("SHARED_TEST", ((0, 1), (1, 2)), (0, 0), (2, 1))
    rng = np.random.default_rng(42)
    target = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n); target /= np.linalg.norm(target)
    parameters = random_parameters(spec, 43)
    numpy_state = analyze_batch(target, parameters, spec, n)[0]
    theta = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
    torch_target = torch.tensor(target, dtype=torch.complex128)
    torch_state = analyze_state_torch(torch_target, theta, spec, n)
    state_error = float(np.max(np.abs(numpy_state - torch_state.detach().numpy())))

    weights_torch = hamming_weights_torch(n, torch.device("cpu"))
    loss = exact_objective(torch_target, theta, spec, n, weights_torch); loss.backward(); gradient = theta.grad.detach().numpy()
    epsilon = 1e-6; tested = [0, 3, 7, 11, 14]; finite_errors = []
    weights_numpy = hamming_weights(n)
    for index in tested:
        plus = parameters.copy(); minus = parameters.copy(); plus[index] += epsilon; minus[index] -= epsilon
        plus_cost = exact_costs(analyze_batch(target, plus, spec, n), weights_numpy)[0]
        minus_cost = exact_costs(analyze_batch(target, minus, spec, n), weights_numpy)[0]
        numerical = (plus_cost - minus_cost) / (2 * epsilon)
        finite_errors.append(abs(numerical - gradient[index]))

    checks = {
        "torch_and_numpy_states_match": bool(state_error < 1e-11),
        "shared_parameter_gradient_matches_finite_difference": bool(max(finite_errors) < 2e-6),
        "gradient_is_finite": bool(np.isfinite(gradient).all()),
    }
    report = {"all_checks_passed": bool(all(checks.values())), "checks": checks, "maximum_state_error": float(state_error), "maximum_gradient_error": float(max(finite_errors)), "torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available())}
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
