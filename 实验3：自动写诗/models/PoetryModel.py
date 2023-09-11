from torch import nn
from .BasicModule import BasicModule






class PoetryModel(BasicModule):
    def __init__(self,vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.modelName="PoetryModel"
        self.embeddings=nn.Embedding(vocab_size,embedding_dim)
        self.hidden_dim=hidden_dim
        # 这稍微有点不一样
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, vocab_size)
        )
    def forward(self,input,hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input)
        # [ seq_len,batch] => [seq_len, batch, embedding_dim]
        output, hidden = self.lstm(embeds, (h_0, c_0))# output的shape为[seq_len,batch,hidden_dim]
        output = self.fc(output.view(seq_len * batch_size, -1))# output的shape为[seq_len*batch,vocab_size]
        return output, hidden 