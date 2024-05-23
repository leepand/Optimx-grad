from optimxgrad import nn
from optimxgrad.nn import functional as F

ACTIVATION_MAP = {
    "tanh": F.tanh,
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "softmax": F.softmax,
}

LOSS_TYPES = {
    "mse": F.MSELoss,
    "bce": F.BCELoss,
    "nll": F.NLLLoss,
}
