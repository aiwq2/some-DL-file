import os
from torch.utils.data import Dataset
import torch







class CommentDataSet(Dataset):
    def __init__(self,path,word2id) -> None:
        super().__init__()
        self.path=path
        self.word2id=word2id
        self.datas,self.labels=self.get_dataAndLable()


    def __getitem__(self, index):
        return self.datas[index],self.labels[index]
    

    def __len__(self):
        return len(self.datas)
    
    def get_dataAndLable(self):
        datas=[]
        labels=[]
        with open(self.path,encoding='utf-8') as f:
            for line in f.readlines():
                if line[0]!='0' and line[0]!='1':
                    continue
                label = torch.tensor(int(line[0]), dtype=torch.int64)
                labels.append(label)
                line_data=[]
                for word in line.strip().split()[1:]:
                    try:
                        index = self.word2id[word]
                    except BaseException:
                        index = 0
                    line_data.append(index)
                datas.append(line_data)
            return datas,labels

# 将batch中的数据对齐
def mycollate_fn(data):
    #step1: 分离data、label
    data.sort(key=lambda x: len(x[0]), reverse=True)
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])

    #step2: 大于75截断、小于75补0
    padded_datas = []
    for data in input_data:
        if len(data) >= 75:
            padded_data = data[:75]
        else:
            padded_data = data
            while (len(padded_data) < 75):
                padded_data.append(0)
        padded_datas.append(padded_data)

    #step3: label、data转为tensor
    label_data = torch.tensor(label_data)
    padded_datas = torch.tensor(padded_datas, dtype=torch.int64)
    return padded_datas, label_data