"""
Optimize kernel hyperparameters with GPflow and print the result (fitted
hyperparameters + per-iteration loss) as one JSON line prefixed with
RESULT_JSON:, so compare.py can pick it out of the surrounding TensorFlow log
output.
"""

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import gpflow

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../../data/data_1024"
TRAIN_SIZE = 512
N_REG = 8
OPT_ITER = 300

gpflow.config.set_default_float(np.float64)


def generate_regressor(x_original, n_regressors):
    x_padded = np.pad(x_original, pad_width=(n_regressors - 1, 0), mode="constant")
    return np.array([x_padded[i : i + n_regressors] for i in range(len(x_original))])


x_train_in = np.loadtxt(DATA_DIR / "training_input.txt", dtype="d")[:TRAIN_SIZE]
X_train = generate_regressor(x_train_in, N_REG).astype("d")
Y_train = np.loadtxt(DATA_DIR / "training_output.txt", dtype="d")[:TRAIN_SIZE, None]

model = gpflow.models.GPR(
    (X_train, Y_train),
    kernel=gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=1.0),
    noise_variance=0.1,
)

opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        loss = model.training_loss()
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


losses = []
for _ in range(OPT_ITER):
    # Divide by N: GPflow's training_loss() is the raw, unnormalized log
    # marginal likelihood, whereas GPRat/GPyTorch both normalize by the
    # number of training points. This only rescales the loss for reporting;
    # it doesn't change the optimization trajectory, since Adam's update is
    # invariant to a constant multiplicative rescaling of the gradient.
    losses.append(float(optimization_step().numpy()) / TRAIN_SIZE)

print(
    "RESULT_JSON:"
    + json.dumps(
        {
            "lengthscale": float(model.kernel.lengthscales.numpy()),
            "variance": float(model.kernel.variance.numpy()),
            "noise": float(model.likelihood.variance.numpy()),
            "losses": losses,
        }
    )
)
