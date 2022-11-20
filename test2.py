from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from PIL import Image

writer = SummaryWriter("logs")
img_path = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\pytorch\\data\\train\\ants_image\\0013035.jpg'
"""img = cv2.imread(img_path)"""
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array.shape)

writer.add_image("test",img_array,1,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()