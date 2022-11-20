import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Mine(nn.Module):
    def __init__(self) -> None:
        super(Mine,self).__init__()
        """self.cov1 = Conv2d(3,32,5,padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.cov2 = Conv2d(32,32,5,padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.cov3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.Linear1 = Linear(1024,64)
        self.Linear2 = Linear(64,10)"""

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        """x = self.cov1(x)
        x = self.maxpool1(x)
        x = self.cov2(x)
        x = self.maxpool2(x)
        x = self.cov3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = self.Linear2(x)"""
        return x

Mine = Mine()
print(Mine)

input = torch.ones((64,3,32,32))
output = Mine(input)
print(output.shape)

writer = SummaryWriter("../Seq_logs")
writer.add_graph(Mine,input)
writer.close()