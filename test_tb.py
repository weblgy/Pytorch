from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "C:/Users/Administrator/PycharmProjects/PythonProject/learn_pytorch/dataset/train/ants_image/5650366_e22b7e1065.jpg"
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)

writer.add_image("ants_image",image_array,2,dataformats="HWC")
writer.close()
for i in range(100):
    writer.add_scalar("y=x",2*i,i)
writer.close()
