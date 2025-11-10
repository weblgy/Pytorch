import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from  model import *

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=torchvision.transforms.ToTensor())

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# 利用Dataloader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
net = Net()
net = net.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epochs = 100

# 添加tensorboard
writer = SummaryWriter('./logs_train')

for i in range(epochs):
    print("-------第 {} 轮训练开始--------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_function(outputs, targets)

        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{} ,loss: {}".format(total_train_step,loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(net,"net_{}.pth".format(i))
    # torch.save(net.state_dict(), './model.pth')
    print("模型已保存")

writer.close()

