import torch
import torchvision.datasets as datasets
from torchvision import transforms,models
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import os
from read_data import CatDog
from model_net.VGG16 import VGG16
from model_net.Resnet50 import ResNet50
import time

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

def sameNumber(y_hat, y):
    '''
    返回预测值与真实值相等的个数
    '''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def display_imginfo(train_data,validate_data):
    print("length of train_data is {}".format(len(train_data)))
    print("length of validate_data is {}".format(len(validate_data)))
    img,target=train_data[0]
    print("the shape of image is {}".format(img.shape))
    print("target is {}".format(target))
    img_display=np.array(img).transpose(1,2,0)
    print(img_display.shape)
    plt.imshow(img_display)
    plt.show()

def read_data(root,batch_size):
    data_train=CatDog(os.path.join(root,'train'),is_train=True)
    data_test=CatDog(os.path.join(root,'test'),is_train=False)
    # 将训练数据七三开为训练数据和验证数据
    train_data, validate_data = data.random_split(data_train, [int(0.7 * len(data_train)), int(0.3 * len(data_train))]) 
    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_data,batch_size=batch_size, shuffle=True)
    print("len data_train:{},data_test:{}".format(len(data_train),len(data_test)))
    test_loader =DataLoader( data_test,batch_size=batch_size, shuffle=False)
    # display_imginfo(data_train,data_test)
    return train_loader,validate_loader,test_loader

def validate(epoch,model,validate_loader,loss_func,device):
    model.eval()
    val_accuracy = 0
    total_num=0
    with torch.no_grad():
        for _,(imgs,targets) in enumerate(validate_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            val_accuracy = val_accuracy + accuracy
            total_num+=targets.shape[0]
    total_accuracy=val_accuracy/total_num
    print("整体验证集上的准确率: {:.4f}".format(total_accuracy))
    writer.add_scalar("val_accuracy", total_accuracy, epoch+1)
    if total_accuracy > 0.95:
            # model.save()
            print("该模型是在第{}个epoch取得95%以上的验证准确率, 准确率为：{:.4f}".format(epoch,  total_accuracy))
    return total_accuracy

def train(model,batch_size,epoch_num,device,learning_rate,writer,pre_load_path=None):
    """
    描述：训练模型并使用验证数据集验证
    """
    # def init(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)
    # model.apply(init)
    if pre_load_path:
        model.load(pre_load_path)
    model.to(device)

    #step2: 训练数据与验证数据
    data = CatDog(root=r'dataset\train', is_train=True)
    train_data, validate_data = Data.random_split(data, [int(0.7 * len(data)), int(0.3 * len(data))])
    print("len train_data {},validate_data {}".format(len(train_data),len(validate_data)))
    train_loader = Data.DataLoader(train_data, batch_size, shuffle=True)
    validate_loader = Data.DataLoader(validate_data, batch_size, shuffle=True)

    # 定义优化函数
    # optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        # 损失函数
    loss_func=nn.CrossEntropyLoss()
    loss_func.to(device)
    #开始训练
    for epoch in range(epoch_num):
        print("-------第 {} 轮训练开始-------".format(epoch+1))
        model.train()
        train_accuracy = 0
        metric = Accumulator(2)
        for step,(imgs,targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs=model(imgs)
            # print("outputs is {}".format(outputs))
            loss=loss_func(outputs,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (outputs.argmax(1) == targets).sum()
            train_accuracy=train_accuracy+accuracy
            with torch.no_grad():
                metric.add(loss * imgs.shape[0], imgs.shape[0])
            if step % 50 == 0:
                train_loss = metric[0] / metric[1]
                train_acc=train_accuracy/metric[1]
                print('epoch: {:d}, 训练次数: {:d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, step, train_loss, train_acc))               
        # 测试步骤开始
        scheduler.step(train_loss)
        train_total_accuracy=train_accuracy/metric[1]
        print("在epoch={}下，整体训练集上的Loss: {:.4f}".format(epoch,train_loss))
        print("在epoch={}下，整体训练集上的准确率: {:.4f}".format(epoch,train_total_accuracy))
        writer.add_scalar("train_loss", train_loss, epoch+1)
        writer.add_scalar("train_accuracy", train_total_accuracy, epoch+1)
        val_acc=validate(epoch,model,validate_loader,loss_func,device)
        if (epoch+1)%20==0:
            model.save()
            print("模型在epoch为{}时保存了一次，此时验证集准确率为{:.4f}，测试集准确率为{:.4f}，时间为{}".format(epoch,train_loss,val_acc,time.strftime('%m%d_%H_%M.pth')))
    return model

def test(batch_size,model,device,writer):
    #step1: 测试数据
    test_data = CatDog(root=r'dataset\train', is_train=False)
    test_loader = Data.DataLoader(test_data, batch_size,shuffle=True)
    print("length of test_data is{}".format(len(test_data)))
    loss_func=nn.CrossEntropyLoss()
    loss_func.to(device)
    model.to(device)
    model.eval()
    val_accuracy = 0
    total_num=0
    with torch.no_grad():
        for i,(imgs,targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            val_accuracy = val_accuracy + accuracy
            total_num+=targets.shape[0]
            writer.add_scalar('Test Loss', loss.item(), i+1)
            writer.add_scalar('Test Accuracy', val_accuracy/total_num, i+1)
    total_accuracy=val_accuracy/total_num
    print("整体测试集上的准确率: {:.4f}".format(total_accuracy))

if __name__ == "__main__":
    #定义存储路径
    root="./dataset"
    # 定义训练中所需要的超参数
    batch_size=10
    learning_rate=0.01
    epoch_num=10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 添加tensorboard
    writer = SummaryWriter("./logs_dogcatRecognition")

    model=VGG16()
    print(model.state_dict())

    #使用vgg预训练模型只更新全连接层参数训练并测试,train函数中就不用apply(init)了
    # model = models.vgg16(pretrained=True)
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    # model.classifier = nn.Sequential(
    #     nn.Linear(512 * 7 * 7, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(512, 2),
    # )

    #使用resnet50预训练模型训练并测试
    # model = models.resnet50(pretrained=True)
    # numFit = model.fc.in_features
    # model.fc = nn.Linear(numFit, 2)

    # train_loader,validate_loader,test_loader=read_data(root,batch_size)
    # trained_model=train(model,batch_size,epoch_num,device,learning_rate,writer)
    # test(batch_size,trained_model,device,writer)
    writer.close()
