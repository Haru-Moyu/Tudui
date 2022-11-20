import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset_Linear",transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Mine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Linear1 = Linear(196608,10)

    def forward(self,input):
        self.Linear1(input)
        return output

Mine = Mine()

for data in dataloader:
    imgs,target = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = Mine(output)
    print(output.shape)