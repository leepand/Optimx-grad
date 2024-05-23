from optimxgrad.engine import Tensor, no_grad
from optimxgrad import nn, optim
from optimxgrad.nn import functional as F


class Model(nn.Module):
    def __init__(self, n_in, n_out, device):
        super().__init__(device)
        self.l1 = nn.Linear(n_in, 6, device=device)
        self.l2 = nn.Linear(6, 9, device=device)
        self.l3 = nn.Linear(9, n_out, device=device)

    def __call__(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)  # F.linear(self.l3(x))

        return x
