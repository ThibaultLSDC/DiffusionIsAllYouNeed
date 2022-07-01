from typing import Tuple
from torch.nn import Module, Conv2d, Tanh, Sequential, ReLU, BatchNorm2d


class BaseNet(Module):
    def __init__(self) -> None:
        super(BaseNet, self).__init__()

        self.core = Sequential(
            Conv2d(1, 512, 3, 1, 1),
            ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            Tanh()
        )

    def forward(self, input, t):
        x = self.core(input)
        return x


class ResBlock(Module):
    def __init__(self, features: list[int]) -> None:
        super(ResBlock, self).__init__()

        in_features = features[0]
        out_features = features[1]

        self.core = Sequential(
            Conv2d(in_features, out_features, 3, 1, 1),
            BatchNorm2d(out_features),
            ReLU(),
            Conv2d(out_features, out_features, 3, 1, 1),
            BatchNorm2d(out_features)
        )

        self.act = ReLU()
    
    def forward(self, input):
        x = self.core(input)
        return self.act(x + input)


class AttentionBlock(Module):
    def __init__(self, patches: int) -> None:
        super().__init__()