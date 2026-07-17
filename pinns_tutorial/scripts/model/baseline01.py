import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        # x and t is input
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


if __name__ == "__main__":
    model = PINN()
    print(model)