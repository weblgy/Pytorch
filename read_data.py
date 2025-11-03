import os

from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.image_paths = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_item_path = os.path.join(self.root_dir,self.label_dir,image_name)
        img = Image.open(image_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_paths)

root_dir = "C:/Users/Administrator/PycharmProjects/PythonProject/learn_pytorch/dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyDataset(root_dir,ants_label_dir)
bees_dataset = MyDataset(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset

