"""
Run gprat_optimize.py / gpflow_optimize.py / gpytorch_optimize.py, each under
its own interpreter (they depend on mutually incompatible environments), and
compare the fitted kernel hyperparameters and loss trajectory across all
three.

All three start from the same hyperparameters and use the same Adam settings
(lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8) for the same number of
iterations. Their optimizers are otherwise independent implementations
(GPRat: hand-written C++ Adam over analytic gradients; GPflow/GPyTorch:
autodiff + framework Adam), so agreement here demonstrates convergence to the
same optimum, not shared code.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BACKENDS = {
    "gprat": [sys.executable, str(SCRIPT_DIR / "gprat_optimize.py")],
    "gpflow": [
        str(SCRIPT_DIR / "../gpflow_reference/gpflow_cpu_env/bin/python"),
        str(SCRIPT_DIR / "gpflow_optimize.py"),
    ],
    "gpytorch": [
        str(SCRIPT_DIR / "../gpytorch_reference/gpytorch_cpu_env/bin/python"),
        str(SCRIPT_DIR / "gpytorch_optimize.py"),
    ],
}

MARKER = "RESULT_JSON:"
CHECKPOINTS = [0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, -1]


def run_backend(name, cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
    for line in result.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    raise RuntimeError(
        f"{name}: no {MARKER} line found in output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def main():
    results = {name: run_backend(name, cmd) for name, cmd in BACKENDS.items()}

    n_iter = len(results["gprat"]["losses"])
    for name, r in results.items():
        assert len(r["losses"]) == n_iter, f"{name}: expected {n_iter} loss values, got {len(r['losses'])}"

    print("Loss trajectory (per-sample-normalized negative log marginal likelihood):")
    print(f"  {'iter':>5} {'gprat':>14} {'gpflow':>14} {'gpytorch':>14}")
    for i in CHECKPOINTS:
        idx = i if i >= 0 else n_iter + i
        print(
            f"  {idx:>5} "
            f"{results['gprat']['losses'][i]:>14.6f} "
            f"{results['gpflow']['losses'][i]:>14.6f} "
            f"{results['gpytorch']['losses'][i]:>14.6f}"
        )

    print("\nFitted hyperparameters:")
    rtol = 1e-2
    ok = True
    for field in ("lengthscale", "variance", "noise"):
        print(f"{field}:")
        for a, b in [("gprat", "gpflow"), ("gprat", "gpytorch"), ("gpflow", "gpytorch")]:
            va, vb = results[a][field], results[b][field]
            rel_diff = abs(va - vb) / max(abs(va), abs(vb), 1e-12)
            print(f"  {a:9s} vs {b:9s}: {va:.6f} vs {vb:.6f}  (rel_diff={rel_diff:.2e})")
            ok &= rel_diff < rtol

    verdict = "PASS: fitted hyperparameters agree" if ok else "FAIL: fitted hyperparameters do not agree"
    print(f"\n{verdict} within rtol={rtol:.0e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
