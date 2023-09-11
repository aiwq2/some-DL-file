import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class TextCNN(BasicModule):
    def __init__(self, vocab_size, embedding_dim, filters_num, filter_size, pre_weight=None):
        super(TextCNN, self).__init__()
        self.modelName = 'TextCNN'
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = False
        self.batch_norm=nn.BatchNorm2d(filters_num)
        if pre_weight is not None:
            self.embeddings = self.embeddings.from_pretrained(pre_weight)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filters_num, (size, embedding_dim)) for size in filter_size])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(filters_num * len(filter_size),512),
            nn.ReLU(),
            nn.Dropout()
            )
        self.fc2=nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout()
            )
        self.fc3=nn.Sequential(
            nn.Linear(128,2)
            )

    def forward(self, x):
        '''
        x的size为(batch_size, max_len)
        '''
        x = self.embeddings(x)  #(batch_size, max_len, embedding_dim)
        x = x.unsqueeze(1)      #(batch_size, 1, max_len, embedding_dim)
        x = torch.tensor(x, dtype=torch.float32)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]#(batch_size, filters_num, max_len-filter_size+1)
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]#(batch_size, filters_num)
        x = torch.cat(x, 1)#(batch_size, len(filter_size)*filters_num)
        x = self.dropout(x)
        x = self.fc1(x)#(batch_size, 512)
        x = self.fc2(x)#(batch_size, 128)
        out = self.fc3(x)#(batch_size, 2)
        return out