import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torchvision
import torch

class CACD(data.Dataset):
    def __init__(self, split='train', transforms=None, label_transforms=None):
        list_root = "/home/guyuchao/PycharmProjects/ResearchAging/AgingGan/data/cacd2000-lists"
        data_root = "/home/guyuchao/Dataset/ExperimentDataset/CACD2000-aligned"
        if split == "train":
            self.list_path=os.path.join(list_root,"train.txt")
        else:
            self.list_path = os.path.join(list_root, "test.txt")
        self.images_labels=[]#path
        self.transform=transforms
        self.label_transform=label_transforms
        with open(self.list_path) as fr:
            lines=fr.readlines()
            for line in lines:
                line.strip()
                item=line.split()
                image_label=[]
                image_label.append(os.path.join(data_root,item[0]))
                image_label.append(np.array(item[1],dtype=np.int))
                self.images_labels.append(image_label)

    def __getitem__(self, idx):
        img_path,label=self.images_labels[idx]
        img=Image.open(img_path)

        if self.transform is not None:
            img=self.transform(img)
        if self.label_transform is None:
            label=torch.from_numpy(label)
        return img,label

    def __len__(self):
        return len(self.images_labels)

if __name__=="__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1., 1., 1.]),
    ])

    CACD_dataset=CACD("train",transforms,None)
    train_loader = torch.utils.data.DataLoader(
        dataset=CACD_dataset,
        batch_size=32,
        shuffle=False
    )
    for idx,(img,label) in enumerate(train_loader):
        print(img.size())
        print(label.size())
        break