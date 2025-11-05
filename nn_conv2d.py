import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(3, 6, 3,1,0)
    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
print(net)

writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    imgs,targets = data
    output = net(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)

    # torch.Size([64, 6, 30, 30]) -> [xx ,3,30,30]
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)

    step = step + 1

writer.close()