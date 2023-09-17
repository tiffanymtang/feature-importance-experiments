import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_normal_X
X_PARAMS_DICT = {
    "n": 500,
    "d": 20  # 200
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "s": 5,
    "beta": 1,
    "heritability": 0.4,
    "sigma": None
}

VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"250": int(250 * 1.2),
                         "500": int(500 * 1.2),
                         "1000": int(1000 * 1.2)}}#, "2000": 2000 * 1.2}}