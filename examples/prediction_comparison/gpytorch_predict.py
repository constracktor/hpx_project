"""
Predict with GPyTorch and print the result as one JSON line prefixed with
RESULT_JSON:, so compare.py can pick it out of the surrounding output.
"""

import json
from pathlib import Path

import numpy as np
import torch
import gpytorch

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../../data/data_1024"
TRAIN_SIZE = 512
TEST_SIZE = 64
N_REG = 8

torch.set_default_dtype(torch.float64)


def generate_regressor(x_original, n_regressors):
    x_padded = np.pad(x_original, pad_width=(n_regressors - 1, 0), mode="constant")
    return np.array([x_padded[i : i + n_regressors] for i in range(len(x_original))], dtype="d")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Zero mean to match GPRat/GPflow, which assume a zero-mean prior.
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


x_train_in = np.loadtxt(DATA_DIR / "training_input.txt", dtype="d")[:TRAIN_SIZE]
x_test_in = np.loadtxt(DATA_DIR / "test_input.txt", dtype="d")[:TEST_SIZE]

X_train = torch.from_numpy(generate_regressor(x_train_in, N_REG))
X_test = torch.from_numpy(generate_regressor(x_test_in, N_REG))
Y_train = torch.from_numpy(np.loadtxt(DATA_DIR / "training_output.txt", dtype="d")[:TRAIN_SIZE])

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 0.1
model = ExactGPModel(X_train, Y_train, likelihood)

model.eval()
likelihood.eval()

with (
    torch.no_grad(),
    gpytorch.settings.fast_pred_var(False),
    gpytorch.settings.lazily_evaluate_kernels(False),
    gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False),
):
    f_pred = model(X_test)
    mean = f_pred.mean
    var = f_pred.variance

print("RESULT_JSON:" + json.dumps({"mean": mean.tolist(), "var": var.tolist()}))
