import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./datasets", download=True, train=False, transform=dataset_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0,drop_last=True)

# 测试数据集中的第一张图片及target
img,target = test_data[0]
print(img)
print(target)

# 使用tensorboard展示
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs,targets = data
    writer.add_images("test_data",imgs,step)
    step += 1

writer.close()