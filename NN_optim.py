import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=1)

class Mine(nn.Module):
    def __init__(self) -> None:
        super(Mine,self).__init__()
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
        return x

loss = nn.CrossEntropyLoss()
Mine = Mine()
optim = torch.optim.SGD(Mine.parameters(),lr=0.01)
for epoch in range(20):
    runnimg_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = Mine(imgs)
        res_loss = loss(outputs,targets)
        optim.zero_grad()
        res_loss.backward()
        optim.step()
        runnimg_loss = res_loss + runnimg_loss
    print(runnimg_loss)