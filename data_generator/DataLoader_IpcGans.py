import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
import os
from random import shuffle
import torchvision
import torch
import numpy as np

class CACD(data.Dataset):
    def __init__(self,split="train",transforms=None, label_transforms=None):

        self.split=split

        #define label 128*128 for condition generate image
        list_root = os.path.abspath("./data/cacd2000-lists")
        data_root = "/home/guyuchao/Dataset/ExperimentDataset/CACD2000-aligned"

        self.condition128=[]
        full_one=np.ones((128,128),dtype=np.float32)
        for i in range(5):
            full_zero=np.zeros((128,128,5),dtype=np.float32)
            full_zero[:,:,i]=full_one
            self.condition128.append(full_zero)

        # define label 64*64 for condition discriminate image
        self.condition64 = []
        full_one = np.ones((64, 64),dtype=np.float32)
        for i in range(5):
            full_zero = np.zeros((64, 64, 5),dtype=np.float32)
            full_zero[:, :, i] = full_one
            self.condition64.append(full_zero)

        #define label_pairs
        label_pair_root=os.path.join(list_root,"train_label_pair.txt")
        with open(label_pair_root,'r') as f:
            lines=f.readlines()
        lines=[line.strip() for line in lines]
        shuffle(lines)
        self.label_pairs=[]
        for line in lines:
            label_pair=[]
            items=line.split()
            label_pair.append(int(items[0]))
            label_pair.append(int(items[1]))
            self.label_pairs.append(label_pair)

        #define group_images
        group_lists = [
            os.path.join(list_root, 'train_age_group_0.txt'),
            os.path.join(list_root, 'train_age_group_1.txt'),
            os.path.join(list_root, 'train_age_group_2.txt'),
            os.path.join(list_root, 'train_age_group_3.txt'),
            os.path.join(list_root, 'train_age_group_4.txt')
        ]

        self.label_group_images = []
        for i in range(len(group_lists)):
            with open(group_lists[i], 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            group_images = []
            for l in lines:
                items = l.split()
                group_images.append(os.path.join(data_root, items[0]))
            self.label_group_images.append(group_images)

        #define train.txt
        if self.split is "train":
            self.source_images = []#which use to aging transfer
            with open(os.path.join(list_root, 'train.txt'), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            shuffle(lines)
            for l in lines:
                items = l.split()
                self.source_images.append(os.path.join(data_root, items[0]))
        else:
            self.source_images = []  # which use to aging transfer
            with open(os.path.join(list_root, 'test.txt'), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            shuffle(lines)
            for l in lines:
                items = l.split()
                self.source_images.append(os.path.join(data_root, items[0]))

        #define pointer
        self.train_group_pointer=[0,0,0,0,0]
        self.source_pointer=0
        self.batch_size=32
        self.transforms=transforms
        self.label_transforms=label_transforms

    def __getitem__(self, idx):
        if self.split is "train":
            pair_idx=idx//self.batch_size #a batch train the same pair
            true_label=int(self.label_pairs[pair_idx][0])
            fake_label=int(self.label_pairs[pair_idx][1])

            true_label_128=self.condition128[true_label]
            true_label_64=self.condition64[true_label]
            fake_label_64=self.condition64[fake_label]

            true_label_img=pil_loader(self.label_group_images[true_label][self.train_group_pointer[true_label]]).resize((128,128))
            source_img=pil_loader(self.source_images[self.source_pointer])

            source_img_227=source_img.resize((227,227))
            source_img_128=source_img.resize((128,128))

            if self.train_group_pointer[true_label]<len(self.label_group_images[true_label])-1:
                self.train_group_pointer[true_label]+=1
            else:
                self.train_group_pointer[true_label]=0

            if self.source_pointer<len(self.source_images)-1:
                self.source_pointer+=1
            else:
                self.source_pointer=0

            if self.transforms is not None:
                true_label_img=self.transforms(true_label_img)
                source_img_227=self.transforms(source_img_227)
                source_img_128=self.transforms(source_img_128)

            if self.label_transforms is not None:
                true_label_128=self.label_transforms(true_label_128)
                true_label_64=self.label_transforms(true_label_64)
                fake_label_64=self.label_transforms(fake_label_64)
            #source img 227 : use it to extract face feature
            #source img 128 : use it to generate different age face -> then resize to (227,227) to extract feature, compile with source img 227
            #ture_label_img : img in target age group -> use to train discriminator
            #true_label_128 : use this condition to generate
            #true_label_64 and fake_label_64 : use this condition to discrimination
            #true_label : label

            return source_img_227,source_img_128,true_label_img,\
                   true_label_128,true_label_64,fake_label_64, true_label
        else:
            source_img_128=pil_loader(self.source_images[idx]).resize((128,128))
            if self.transforms is not None:
                source_img_128=self.transforms(source_img_128)
            condition_128_tensor_li=[]
            if self.label_transforms is not None:
                for condition in self.condition128:
                    condition_128_tensor_li.append(self.label_transforms(condition).cuda())
            return source_img_128.cuda(),condition_128_tensor_li

    def __len__(self):
        if self.split is "train":
            return len(self.label_pairs)
        else:
            return len(self.source_images)

if __name__=="__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    label_transforms=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    CACD_dataset=CACD(transforms,label_transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=CACD_dataset,
        batch_size=32,
        shuffle=True
    )
    for idx,\
        (source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, true_label) in enumerate(train_loader):
        print(true_label)
        break
