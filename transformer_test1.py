from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2

img_path = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\pytorch\\data\\train\\ants_image\\0013035.jpg'
img = cv2.imread(img_path)

writer = SummaryWriter('logs2')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

#print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()