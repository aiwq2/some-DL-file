import os
import torch
import torch.optim as optim

path=os.path.join(".","models_save","TextCNN_0412_10_31.pth")
model=torch.load(path)
# print(model)
# for k in model:
#     print(k)
#     print(model[k].shape)


optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
for var_name in optimizer.state_dict():
    print(var_name,'\t',optimizer.state_dict()[var_name])
