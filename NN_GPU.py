import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from model import *

class Mine(nn.Module):
    def __init__(self):
        super(Mine,self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

Mine = Mine()
Mine.cuda()

writer = SummaryWriter("../train_logs")

train_data = torchvision.datasets.CIFAR10(root="../train_data", transform=torchvision.transforms.ToTensor(),
                                          train=True, download=True)
test_data = torchvision.datasets.CIFAR10(root="../test_data", transform=torchvision.transforms.ToTensor(),
                                         train=False, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


loss_fuction = nn.CrossEntropyLoss()
loss_fuction = loss_fuction.cuda()

learning_rate = 0.01
optimizer = torch.optim.SGD(Mine.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

start_time = time.time()

for i in range(epoch):
    Mine.train()
    print("第{}轮开始".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = Mine(imgs)
        loss = loss_fuction(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print('训练次数：{}，Loss：{}'.format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_accuracy = 0
    Mine.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = Mine(imgs)
            loss = loss_fuction(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    print("整体测试集上的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(Mine.state_dict(), "Mine_{}.pth".format(i))
    print('save sucessed')

writer.close()
