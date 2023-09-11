import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CatDog(Dataset):

    def __init__(self, root, is_train=True):
        # 原始数据集中的图片数量太多，这里从原始训练数据集中选择8000张打乱的猫狗图片，从原始训练数据集中选择2000张作为测试集
        pet_num_train=4000
        pet_num_test=1000
        base_cat=8000
        base_dog=12600
        self.imgs_temp = [os.path.join(root, img) for img in os.listdir(root)]
        self.imgs=[]
        if is_train:
            self.imgs.extend(self.imgs_temp[:pet_num_train])
            self.imgs.extend(self.imgs_temp[-1*pet_num_train:])
        else:
            self.imgs.extend(self.imgs_temp[base_cat:base_cat+pet_num_test])
            self.imgs.extend(self.imgs_temp[base_dog:base_dog+pet_num_test])
        self.train = is_train
        if not is_train:
            self.transform=transforms.Compose([
                            transforms.Resize(size=(224,224)),
                            transforms.CenterCrop(size=(224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        else:
            self.transform=transforms.Compose([
                        transforms.Resize(size=(256, 256)),
                        transforms.RandomResizedCrop(size=(224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label =  1 if 'dog' in img_path.split('.')[0] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)