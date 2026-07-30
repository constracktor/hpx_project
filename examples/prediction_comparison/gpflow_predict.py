"""
Predict with GPflow and print the result as one JSON line prefixed with
RESULT_JSON:, so compare.py can pick it out of the surrounding
TensorFlow log output.
"""

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import gpflow

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../../data/data_1024"
TRAIN_SIZE = 512
TEST_SIZE = 64
N_REG = 8

gpflow.config.set_default_float(np.float64)


def generate_regressor(x_original, n_regressors):
    x_padded = np.pad(x_original, pad_width=(n_regressors - 1, 0), mode="constant")
    return np.array([x_padded[i : i + n_regressors] for i in range(len(x_original))])


x_train_in = np.loadtxt(DATA_DIR / "training_input.txt", dtype="d")[:TRAIN_SIZE]
x_test_in = np.loadtxt(DATA_DIR / "test_input.txt", dtype="d")[:TEST_SIZE]

X_train = generate_regressor(x_train_in, N_REG).astype("d")
X_test = generate_regressor(x_test_in, N_REG).astype("d")
Y_train = np.loadtxt(DATA_DIR / "training_output.txt", dtype="d")[:TRAIN_SIZE, None]

model = gpflow.models.GPR(
    (X_train, Y_train),
    kernel=gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=1.0),
    noise_variance=0.1,
)

mean, var = model.predict_f(X_test)

print(
    "RESULT_JSON:"
    + json.dumps({"mean": mean.numpy().flatten().tolist(), "var": var.numpy().flatten().tolist()})
)
