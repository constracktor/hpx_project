"""
Optimize kernel hyperparameters with GPRat and print the result (fitted
hyperparameters + per-iteration loss) as one JSON line prefixed with
RESULT_JSON:, so compare.py can pick it out of the surrounding HPX/APEX
startup output.
"""

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))
import gprat  # noqa: E402

DATA_DIR = SCRIPT_DIR / "../../data/data_1024"
TRAIN_SIZE = 512
N_REG = 8
N_TILES = 4
OPT_ITER = 300
KERNEL_PARAMS = [1.0, 1.0, 0.1]  # lengthscale, vertical_lengthscale, noise_variance

train_in = gprat.GP_data(str(DATA_DIR / "training_input.txt"), TRAIN_SIZE, N_REG)
# n_reg=1 -> no offset: unlike the input, the training output has no
# lookahead padding requirement.
train_out = gprat.GP_data(str(DATA_DIR / "training_output.txt"), TRAIN_SIZE, 1)

n_tile_size = gprat.compute_train_tile_size(TRAIN_SIZE, N_TILES)

gp = gprat.GP(
    train_in.data,
    train_out.data,
    N_TILES,
    n_tile_size,
    kernel_params=KERNEL_PARAMS,
    n_reg=N_REG,
    trainable=[True, True, True],
)

gprat.start_hpx(sys.argv, 2)
hpar = gprat.AdamParams(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, opt_iter=OPT_ITER)
losses = gp.optimize(hpar)
gprat.stop_hpx()

# kernel_params is a bound C++ struct (SEKParams), not convertible to a
# Python value directly, so pull the fitted values back out of __repr__.
m = re.search(
    r"lengthscale=([\d.eE+-]+), vertical_lengthscale=([\d.eE+-]+), noise_variance=([\d.eE+-]+)", repr(gp)
)
lengthscale, variance, noise = (float(g) for g in m.groups())

print(
    "RESULT_JSON:"
    + json.dumps({"lengthscale": lengthscale, "variance": variance, "noise": noise, "losses": list(losses)})
)
