import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib
matplotlib.use('Agg')  # 添加这行，设置后端为Agg


BATCH_SIZE = 2048
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

print("DEVICE:", DEVICE, "BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, out):
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.log_softmax(self.fc5(out), dim=1)
        return out


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("data", is_train, transform=to_tensor,
                     download=True)
    return DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # 使用 PyTorch 的 torch.no_grad() 上下文管理器。这是为了确保在评估模型时不会计算梯度，从而节省内存并提高速度。因为在测试或评估阶段，我们不需要反向传播来更新模型的权重
        for (images, labels) in test_data:
            # print(len(images),len(labels)) # BATCH_SIZE
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            inputs = images.view(-1, 28*28)  # 形状调整为 (-1, 28*28)，满足模型的输入要求
            outputs = net.forward(inputs)  # 预测一批数据
            for i, output in enumerate(outputs):
                predict = torch.argmax(output)
                if predict == labels[i]:
                    n_correct += 1
                # else:
                #     print("predict=", int(predict), "label=", int(labels[i]))
                n_total += 1
    return n_correct / n_total


def prepare():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    # 创建神经网络
    net = Net().to(DEVICE)
    return train_data, test_data, net


def train(train_data, test_data, net):
    # 测试网络的初始精度
    print("initial accuracy:", evaluate(test_data, net))

    # 优化器
    # 这是一个注释，用于说明接下来的代码是关于优化器的设置。
    # 这行代码创建了一个Adam优化器实例。Adam是一种自适应学习率的优化算法，常用于训练深度学习模型。
    # net.parameters()提取了神经网络net的所有参数（即权重和偏置）。
    # lr=0.001设置了学习率为0.001。学习率是优化过程中的一个重要超参数，它决定了在每次更新时参数变化的步长大小。
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练网络
    for epoch in range(1, EPOCHS + 1):
        # 这行代码开始了一个循环，用于遍历训练数据集train_data。
        # train_data通常是一个数据加载器（如torch.utils.data.DataLoader），它会生成一批批的图像（images）和对应的标签（labels）。
        for (images, labels) in train_data:
            # 这行代码将图像和标签数据移动到指定的设备（如GPU）上，以便进行加速计算。
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # 放入GPU
            # 这行代码清除了神经网络net的梯度。在PyTorch中，梯度是累加的，因此在每次新的前向传播和反向传播之前，需要清除旧的梯度。
            optimizer.zero_grad()
            # 这行代码重新整形了图像数据。假设每个图像是28x28像素的，这行代码将其整形为一个长度为28*28的一维向量，并保留批次大小作为第一个维度（由-1表示）。这是神经网络输入层的常见要求。
            inputs = images.view(-1, 28*28)
            # 这行代码通过神经网络net的前向传播函数传递了输入数据，并得到了输出outputs
            outputs = net.forward(inputs)
            # 这行代码计算了网络输出outputs与真实标签labels之间的负对数似然损失（Negative Log Likelihood Loss）。这是一种常用于分类任务的损失函数。
            loss = F.nll_loss(outputs, labels)
            # 这行代码执行了反向传播算法，计算了损失函数关于模型参数的梯度。
            loss.backward()
            # 这行代码使用之前定义的Adam优化器来更新神经网络的参数。它会根据计算出的梯度来调整参数，以最小化损失函数。
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))


def test(test_data, net):
    # 测试
    for (n, (images, _)) in enumerate(test_data):
        if n > 3:
            break
        images = images.to(DEVICE)  # 从CPU放到GPU
        inputs = images[0].view(-1, 28*28)
        outputs = net.forward(inputs)
        predict = torch.argmax(outputs)
        plt.figure(n)
        image = images[0]
        image = image.cpu()  # 从GPU放回CPU
        plt.imshow(image.view(28, 28))
        plt.title("prediction: " + str(int(predict)))
        plt.savefig(f"test_image_{n}.png")  # 保存图像为文件，文件名可以根据需求调整
    # plt.show()  注释掉原来的显示命令


if __name__ == "__main__":
    train_data, test_data, net = prepare()
    if os.path.exists("net.pth"):
        net.load_state_dict(torch.load("net.pth", weights_only=True))
    else:
        train(train_data, test_data, net)

    torch.save(net.state_dict(), "net.pth")
    test(test_data, net)
