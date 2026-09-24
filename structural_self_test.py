#!/usr/bin/env python3
"""Dependency-light structural checks for all fairness protocols."""

from __future__ import annotations

import json

from run_causal_cone_local import load_protocol as load_local, validate as validate_local
from run_gate_exposure import load_protocol as load_gate, validate as validate_gate
from run_noiseless_multistart import load_protocol as load_exact, validate as validate_exact


def main():
    gate, _ = load_gate(); exact, _ = load_exact(); local, _ = load_local()
    reports = {"gate_exposure": validate_gate(gate), "noiseless_multistart": validate_exact(exact), "causal_cone_local": validate_local(local)}
    result = {"all_checks_passed": all(report["all_checks_passed"] for report in reports.values()), "reports": reports}
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
