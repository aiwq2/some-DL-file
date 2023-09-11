from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch
# 导入预测指标计算函数和混淆矩阵计算函数
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from config import Config
from BuildWord import build_word2id,build_word2vec
from models import TextCNN,LSTMModel,TextCNN2,LSTMModel2
from read_data import CommentDataSet,mycollate_fn

def sameNumber(y_hat, y):
    """
    返回预测值与真实值相等的个数
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()

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

def train(model,batch_size, lr, epochs,device,writer,train_path,validate_path,pre_load_path=None):
    def init(m):
        if type(m) in [nn.Linear,nn.Conv2d,nn.Conv1d]:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    model.apply(init)
    if pre_load_path:
        model.load_state_dict(torch.load(pre_load_path))

    model.to(device)

    # 查看model所在设备
    # print(next(model.parameters()).device)
    
    # 制作dataloader
    #训练数据
    traindata = CommentDataSet(train_path,word2id)
    #使用了collate_fn参数，shuffle就不能设置为true了，验证集同理
    trainloader = Data.DataLoader(traindata, batch_size, shuffle=True,collate_fn=mycollate_fn)
    #验证数据
    validatedata = CommentDataSet(validate_path, word2id)
    validateloader = Data.DataLoader(validatedata, batch_size, shuffle=True,collate_fn=mycollate_fn)

    # 19998
    # print("length of traindata is {}".format(len(traindata)))
    # 5629
    # print("length of validatedata is {}".format(len(validatedata)))

    # 定义目标函数与优化器，规定学习率衰减规则
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    metric=Accumulator(3)
    for epoch in range(epochs):
        print("-------第 {} 轮训练开始-------".format(epoch+1))
        model.train()
        hidden=None
        for step,(datas,labels) in enumerate(trainloader):
            datas=datas.to(device)
            labels=labels.to(device)
            # 查看数据和label的device
            # print(datas.device)
            # print(labels.device)
            if model.modelName == 'LSTMModel':
                outputs, hidden = model(datas)
            elif model.modelName == 'TextCNN' :
                outputs = model(datas)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric.add(loss * datas.shape[0], sameNumber(outputs, labels), datas.shape[0])
            train_loss=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]
            if step % 50 == 0:
                print('epoch: {:d}, 训练次数: {:d}, 训练损失: {:.4f}, 训练准确率: {:.4f}'.format(epoch, step, train_loss, train_acc))
        scheduler.step(train_loss)
        print("在epoch={}下，整体训练集上的Loss: {:.4f}".format(epoch,train_loss))
        print("在epoch={}下，整体训练集上的准确率: {:.4f}".format(epoch,train_acc))
        writer.add_scalar('Epoch Loss', train_loss, epoch)
        writer.add_scalar('Accuracy', train_acc, epoch)

        # 验证集进行计算
        validate_acc=validate(epoch,model,validateloader,criterion,device)
        
    #如果训练集和验证集的准确率都达到83%以上，则予以保存
    if train_acc>0.83 and validate_acc>0.83:
        model.save()
        print("模型训练准确率为{}，达到83%以上,现已进行保存".format(train_acc))

    return model

def validate(epoch,model,validateloader,criterion,device):
    metric=Accumulator(2)
    model.eval()
    with torch.no_grad():
        for _,(datas,labels) in enumerate(validateloader):
            datas = datas.to(device)
            labels = labels.to(device)
            if model.modelName == 'LSTMModel':
                outputs, hidden = model(datas)
            elif model.modelName == 'TextCNN':
                outputs = model(datas)
            loss = criterion(outputs, labels)
            metric.add(sameNumber(outputs,labels),datas.shape[0])
    total_accuracy=metric[0]/metric[1]
    print("整体验证集上的准确率: {:.4f}".format(total_accuracy))
    writer.add_scalar("val_accuracy", total_accuracy, epoch)
    return total_accuracy

def test(batch_size,model,device,testpath,writer):
    #step1: 测试数据
    test_data = CommentDataSet(testpath,word2id)
    test_loader = Data.DataLoader(test_data, batch_size,shuffle=True,collate_fn=mycollate_fn)
    # 长度369
    # print("length of test_data is {}".format(len(test_data)))
    loss_func=nn.CrossEntropyLoss()
    loss_func.to(device)
    model.to(device)
    model.eval()
    metric=Accumulator(3)
    with torch.no_grad():
        for i,(datas,labels) in enumerate(test_loader):
            datas = datas.to(device)
            labels =labels.to(device)
            if model.modelName == 'LSTMModel':
                outputs, hidden = model(datas)
            elif model.modelName == 'TextCNN':
                outputs = model(datas)
            loss = loss_func(outputs,labels)
            metric.add(loss*labels.shape[0],sameNumber(outputs,labels),labels.shape[0])
            if i!=0:
                outputs_confusion=torch.cat((outputs_confusion,outputs))
                labels_confusion=torch.cat((labels_confusion,labels))
            else:
                outputs_confusion=outputs
                labels_confusion=labels
    total_loss=metric[0]/metric[2]
    total_accuracy=metric[1]/metric[2]
    cm=confusion_matrix(labels_confusion.cpu(),outputs_confusion.argmax(dim=1).cpu())
    print("整体测试集上的loss为:{:.4f},准确率: {:.4f}".format(total_loss,total_accuracy))
    print("---------------------------------------------")
    # precision（精度）：关注于所有被预测为正（负）的样本中究竟有多少是正（负）。（分母是预测出的数据）
    # recall（召回率）： 关注于所有真实为正（负）的样本有多少被准确预测出来了。（分母是原数据）
    # f1-score：二者均值。
    # supprot：每个标签的出现次数。
    # avg / total行为各列的均值（support列为总和）。
    print("准确率，召回率，f1分数为:")
    print(classification_report(labels_confusion.cpu(),outputs_confusion.argmax(dim=1).cpu()))
    print("---------------------------------------------")
    print("混淆矩阵为:")
    print(cm)
    # 将混淆矩阵以热力图的防线显示
    sns.set()
    ax = plt.subplot()
    # 画热力图
    sns.heatmap(cm, cmap="YlGnBu_r", annot=True,fmt='.0f')
    # 标题
    ax.set_title('confusion matrix')
    # x轴为预测类别
    ax.set_xlabel('predict')
    # y轴实际类别
    ax.set_ylabel('positive or negative')
    plt.savefig("pics/confusion_matrix.jpg")
    plt.show()
    
if __name__ == "__main__":
    # 引入配置类
    config=Config()

    # 构建词汇表并存储，形如{word: id}
    word2id,id2word=build_word2id(config.trainpath, config.validatepath, config.testpath) 
    # 共59290个不同词语
    # print(len(word2id))
    # print(len(id2word))

    # 基于预训练好的 word2vec 构建训练语料中所含词语的 word2vec，形如{id:vecs}
    id2vecs = build_word2vec(config.word2vec_pretrained, word2id, save_to_path=None)
    # 每个词向量是五十维
    # print(word2vecs[5].shape)

    # model=TextCNN(config.vocab_size, config.embedding_dim, filters_num=128, filter_size=[1,3,5,7,9], pre_weight=id2vecs)
    # model=TextCNN2(config.vocab_size, config.embedding_dim,kernel_sizes=[64,64,64],num_channels=[3,4,5], pre_weight=id2vecs)
    pre_load_path=r'models_save\TextCNN_0416_11_57.pth'
    # 使用LSTMMOdel进行计算
    # model = LSTMModel(config.embedding_dim, config.hidden_dim, pre_weight=id2vecs)
    model = LSTMModel2(config.vocab_size,config.embedding_dim, config.hidden_dim, num_layers=2,pre_weight=id2vecs)

    # 添加tensorboard
    writer = SummaryWriter("./logs_SentimentAnalysis")

    # 训练模型
    trained_model = train(model, config.batch_size, config.lr, config.epochs, config.device,writer,config.trainpath,config.validatepath)
    test(config.batch_size,trained_model,config.device,config.testpath,writer)
    writer.close()