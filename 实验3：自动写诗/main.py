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
from models import PoetryModel
import torch.optim as optim
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
    
def prepareData(filename,batch_size):
    dataset = np.load(filename,allow_pickle=True)
    # 查看该压缩文件里的所有文件名
    # print(dataset.files)
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    # 验证得到voc_size为8293
    # print('len(ix2word):',len(ix2word))
    # print(len(word2ix))
    # print(ix2word[8292])# 编号为8292的词为<s>
    # print(word2ix['<START>'])# START的编号为8291,<EOP>编号为8290  
    data = torch.from_numpy(data)
    # print('data.shape:',data.shape)
    # print(data[:3])
    # # 尝试打印一下第一首词是什么
    # words=""
    # for idx in data[1]:
    #     if idx!=8292:
    #         words+=ix2word[idx.item()]
    # print("words is {}".format(words))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # print(len(dataloader))
    return dataloader, ix2word, word2ix

def train(model, filename, batch_size, lr, epochs, device, writer,pre_model_path=None):
    if pre_model_path:
        model.load(pre_model_path)
        print("已加载预训练模型，模型加载路径为",pre_model_path)
    model.to(device)

    dataloader, ix2word, word2ix = prepareData(filename, batch_size)
    
    # 定义优化函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    # 定义损失函数
    criterion  = nn.CrossEntropyLoss()
    criterion.to(device)
    metric = Accumulator(2)
    for epoch in range(epochs):
        print("-------第 {} 轮训练开始-------".format(epoch+1))
        model.train()
        for i, data in enumerate(dataloader):
            # 因为在LSTM里面，我们的batch_first参数为false，所以我们的输入要设置为[ seq_len,batch]
            data = data.long().transpose(1, 0).contiguous() 
            data = data.to(device)
            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)# output的shape为[(seq_len-1)*batch,vocab_size]
            loss = criterion(output, target.view(-1))# target的shape为[(seq_len-1)*batch]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * data.shape[1], data.shape[1])
            train_loss = metric[0] / metric[1]
            if i % 100 == 0:
                print('epoch: {:d}, 训练次数: {:d}, 训练损失: {:.4f}'.format(epoch, i, train_loss))
        scheduler.step(train_loss)

    #step5: 迭代结束保存模型
    model.save(epochs)
    return model

def generate(model, filename, device, start_words, max_gen_len, prefix_words):
    #step1: 设置模型参数
    _, ix2word, word2ix = prepareData(filename, 1)
    model.to(device)
    results = list(start_words)
    start_word_len = len(start_words)

    #step2: 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None
    model.eval()
    #step3: 生成唐诗
    with torch.no_grad():
        for i in range(max_gen_len):
            output, hidden = model(input, hidden)
            # 读取第一句
            if i < start_word_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
            # 生成后面的句子
            else:
                # topk也可以用argmax来计算
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)
            # 结束标志
            if w == '<EOP>':
                del results[-1]
                break

    #step4: 返回结果
    return results

def generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic):
     #step1: 设置模型参数
    _, ix2word, word2ix = prepareData(filename, 1)
    model.to(device)
    results = []
    start_word_len = len(start_words_acrostic)
    index = 0
    pre_word = '<START>'

    #step2: 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    #step3: 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        if (pre_word in {'。', '！', '<START>'}):
            if index == start_word_len:
                break
            else:
                w = start_words_acrostic[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    # step4: 返回结果
    return results

if __name__ == "__main__":
    #定义存储路径
    filename = r'dataset\tang.npz'

    # 定义训练中所需要的超参数
    batch_size = 64
    lr = 0.001
    epochs = 50
    vocab_size = 8293
    embedding_dim = 128
    hidden_dim = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 添加tensorboard
    writer = SummaryWriter("./logs_autoPoetry")

    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    # 已经训练了二十遍的模型，再训练20遍
    # pre_model_path=r"models_save\_class 'models.PoetryModel.PoetryModel'__0406_04_44.pth"
    pre_model_path=r"models_save\PoetryModel_0407_09_26.pth"
    model=train(model, filename, batch_size, lr, epochs, device, writer)
    # model=train(model, filename, batch_size, lr, epochs, device, writer,pre_model_path)

    #给定前一句生成后一句
    # start_words = ["飞流直下三千尺","月落乌啼霜满天","拣尽寒枝不肯歇","春宵苦短日高起","只恐双溪舴艋舟","衣带渐宽终不悔"]
    # # model.load(pre_model_path)
    # # # model.load(r"models_save\PoetryModel_0407_09_26.pth")
    # max_gen_len = 32
    # prefix_words = None
    # for start_seq in start_words:
    #     poetry = ''
    #     result = generate(model, filename, device, start_seq, max_gen_len, prefix_words)
    #     for word in result:
    #         poetry += word
    #         if word == '。' or word == '!' or word=='?':
    #             poetry += '\n'
    #     print("{}来进行续写的诗句为\n{}".format(start_seq,poetry))
    
    #生成藏头诗
    start_words_acrostic = ['计算机视觉','自然语言处理','大预言模型','深度学习']
    model.load(r"models_save\PoetryModel_0407_10_26.pth")
    max_gen_len_acrostic = 128
    prefix_words_acrostic = None
    for acrostic in start_words_acrostic:
        poetry = ''
        result = generate_acrostic(model, filename, device, acrostic, max_gen_len_acrostic, prefix_words_acrostic)
        for word in result:
            poetry += word
            if word == '。' or word == '!' or word=='?':
                poetry += '\n'
        print("{}作为藏头词来进行构造的诗句为\n{}".format(acrostic,poetry))

    writer.close()