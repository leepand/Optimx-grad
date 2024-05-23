from optimxgrad.engine import Tensor, no_grad
from optimxgrad import nn, optim
from optimxgrad.nn import functional as F
from ..common.utils import ACTIVATION_MAP, LOSS_TYPES


class Model(nn.Module):
    def __init__(
        self, n_in, layers, n_out, activ="tanh", last_layer="iden", device="cpu"
    ):
        super().__init__(device)
        self.activ = activ
        self.last_layer = last_layer
        self.layers = []
        for i, hidden_dim in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(n_in, hidden_dim, device=device))
            else:
                if i == len(layers) - 1:
                    self.layers.append(nn.Linear(hidden_dim, n_out, device=device))
                else:
                    self.layers.append(
                        nn.Linear(layers[i - 1], hidden_dim, device=device)
                    )
            # self.l1 = nn.Linear(n_in, 6, device=device)
            # self.l2 = nn.Linear(6, 9, device=device)
            # self.l3 = nn.Linear(9, n_out, device=device)

    def __call__(self, x):
        for i, layer_fn in enumerate(self.layers):
            if i == len(self.layers) - 1:
                if self.last_layer == "iden":
                    x = layer_fn(x)
                else:
                    x = ACTIVATION_MAP[self.activ](layer_fn(x))
            else:
                x = ACTIVATION_MAP[self.activ](layer_fn(x))

        return x


class MLP:
    def __init__(
        self,
        n_in,
        layers,
        n_out,
        activ="tanh",
        last_layer="iden",
        device="cpu",
        lr=0.009,
    ) -> None:
        self.model = Model(
            n_in, layers, n_out, activ=activ, last_layer=last_layer, device=device
        )
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, device=device)

    def learn(self, x, y, loss_type="mse"):
        X = Tensor(x).to(self.device)
        y = Tensor(y).to(self.device)
        print(X.shape, y.shape)
        yHat = self.model(X)
        loss = LOSS_TYPES[loss_type](y, yHat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        X = Tensor(x).to(self.device)
        return self.model(X)
