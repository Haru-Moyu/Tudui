import torchvision

#train_data = torchvision.datasets.ImageNet("../data_image_net",split = "train",download=True,
                                           #transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_False = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)

print(vgg16_True)

vgg16_True.add_module("add_Linear",nn.Linear(1000,10))
print(vgg16_True)