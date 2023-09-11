import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# 定义训练中所需要的超参数
batch_size=256
num_workers=4
learning_rate=0.01
epoch_num=30

# 添加tensorboard
writer = SummaryWriter("./logs_writtenRecognition")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.MNIST("./data", train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.MNIST("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
train_data_size=len(train_data)
test_data_size=len(test_data)

print("length of train_data is {}".format(train_data_size))
print("length of test_data is {}".format(test_data_size))
img,target=train_data[0]
print("the shape of image is {}".format(img.shape))
print("target is {}".format(target))
img_display=np.array(img).squeeze()
plt.imshow(img_display)
plt.show()

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

def init(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

class Accumulator():
    '''
    构建n列变量，每列累加
    '''
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Sequential( # 输入图像（1，28，28）
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2), # 输出图像（16，28，28）
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2), # 输出图像（16，14，14）
        )
        self.conv2=nn.Sequential( # 输入图像（16，14，14）
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2), # 输出图像（32，14，14）
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2), # 输出图像（32，7，7）
        )
        # self.flatten=nn.Flatten()
        self.out=nn.Linear(32*7*7,10) # 输出为10个类
    def forward(self,input):
        input=self.conv1(input)
        input=self.conv2(input)
        input=input.view(input.size(0),-1)
        output=self.out(input)
        return output


class MyCNN(nn.Module):
    '''
    构建卷积神经网络,与LeNet类似
    '''
    def __init__(self):
        super(MyCNN, self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),#6*28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)#6*14*14
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1), #16*10*10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2) #16*5*5
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def reshape_(self, x):
        return x.reshape(-1, 1, 28, 28)

    def forward(self, x):
        x = self.reshape_(x)
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.FC1(x)
        return x

cnn=CNN()
cnn.to(device)
cnn.apply(init)
# cnn=MyCNN()
# 定义优化函数
# optimizer=torch.optim.Adam(cnn.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# 损失函数
loss_func=nn.CrossEntropyLoss()
loss_func.to(device)

#开始训练
for epoch in range(epoch_num):
    print("-------第 {} 轮训练开始-------".format(epoch+1))
    metric = Accumulator(2)
    cnn.train()
    train_accuracy = 0
    for step ,(imgs,targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs=cnn(imgs)
        print(outputs.shape)
        # print("output shape is {},targets shape is {}".format(output.shape,targets.shape))
        loss=loss_func(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = (outputs.argmax(1) == targets).sum()
        train_accuracy=train_accuracy+accuracy
        metric.add(loss * imgs.shape[0], imgs.shape[0])
        # if step%100==0:
        #     print("训练次数：{}, Loss: {}".format(step, loss.item()))
        #     writer.add_scalar("train_loss", loss.item(), step)
    # 测试步骤开始
    train_loss = metric[0] / metric[1]
    print("整体训练集上的Loss: {}".format(train_loss))
    print("整体训练集上的准确率: {}".format(train_accuracy/train_data_size))
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy/train_data_size, epoch)
    cnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for step,(imgs,targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cnn(imgs)
            loss = loss_func(outputs, targets)
            # total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的准确率: {}".format(total_accuracy/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, epoch)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, epoch)
torch.save(cnn.state_dict,"cnn.pth")
print("模型已保存")
writer.close()
