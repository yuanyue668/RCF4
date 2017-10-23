import os
import torch
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms,utils
from skimage import io, transform

class TrakingDataSet(Dataset):
    "Tracking Data"
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pathDir = os.listdir(self.root_dir)
        self.pathDir.sort()

    def __len__(self):
        return self.pathDir.__len__()-1

    def __getitem__(self,idex):
        file_name = os.path.join(self.root_dir, self.pathDir[idex])
        # gt = np.loadtxt(os.path.join(self.root_dir,'groundtruth.txt'), delimiter=',')
        try:
            gt = np.loadtxt(os.path.join(self.root_dir, 'groundtruth.txt'))
        except Exception, e:
            gt = np.loadtxt(os.path.join(self.root_dir, 'groundtruth.txt'), delimiter=",")

        print(file_name)

        if (file_name.endswith('.txt') == False):
            image = Image.open(file_name)
            print(gt[idex])

            left = round(gt[idex][0],3)
            up = round(gt[idex][1],3)
            w = round(left + gt[idex][2],3)
            h = round(up + gt[idex][3],3)

            box = (left, up, w, h)
            #print(box)

            patch = image.crop(box)
            sample = {'image': image, 'patch': patch}

        if self.transform:
            sample = self.transform(sample)
        return sample

trakingSet = TrakingDataSet("/home/icv/PyTorch/Two_Stream_CNN_RNN/biker")

fig = plt.figure()

i=0
for sample in trakingSet:
    print(sample['image'].size)

    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['patch'])
    i = i+1
    if i == 3:
        plt.show()
        break
