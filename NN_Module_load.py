import torch
import torchvision

model = torch.load("vgg16_method1.pth")

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")
#print(model)

