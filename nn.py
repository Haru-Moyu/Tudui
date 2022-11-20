from torch import nn
import torch

class mine(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

mine = mine()
x = torch.tensor(1.0)
output = mine(x)
print(output)