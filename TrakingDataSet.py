import os
import torch
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms,utils

class TrakingDataSet(Dataset):
    "Tracking Data"
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgfile = os.path.join(self.root_dir, 'img')
        if (os.path.exists(self.imgfile) == False):
            self.imgfile = os.path.join(self.root_dir, 'imgs')
        self.imgpath = os.listdir(self.imgfile)
        self.imgpath.sort()
        self.imgpath = self.imgpath[:-1]

    def __len__(self):
        #not use the last image
        return self.imgpath.__len__()

    def __getitem__(self,idex):
        img_file = os.path.join(self.imgfile,self.imgpath[idex])
        # gt = np.loadtxt(os.path.join(self.root_dir,'groundtruth.txt'), delimiter=',')
        #print(img_file)
        gt_file = os.path.join(self.root_dir, 'groundtruth.txt')
        if(os.path.exists(gt_file) == False):
            gt_file = os.path.join(self.root_dir,'groundtruth_rect.txt')
        try:
            gt = np.loadtxt(gt_file)
        except Exception, e:
            gt = np.loadtxt(gt_file, delimiter=",")


        image = Image.open(img_file)
        if(image.mode == 'L'):
            r,g,b = image,image,image
            image = Image.merge("RGB",(r,g,b))

        left = round(gt[idex][0],3)
        up = round(gt[idex][1],3)
        left_plus_w = round(left + gt[idex][2],3)
        up_plus_h = round(up + gt[idex][3],3)
        box = (left, up, left_plus_w, up_plus_h)

        #convert the values of (x,y,w,h) to (0,1)
        pre_x = round((gt[idex][0]+gt[idex][2]/2)/image.size[0],3)
        pre_y = round((gt[idex][1]+gt[idex][3]/2)/image.size[1],3)
        pre_w = round(gt[idex][2]/image.size[0],3)
        pre_h = round(gt[idex][3]/image.size[1],3)
        coordinate = (pre_x,pre_y,pre_w,pre_h)
        coordinate = torch.FloatTensor(coordinate)

        x = round((gt[idex+1][0]+gt[idex+1][2]/2)/image.size[0], 3)
        y = round((gt[idex+1][1]+gt[idex+1][3]/2)/image.size[1], 3)
        w = round(gt[idex+1][2]/image.size[0], 3)
        h = round(gt[idex+1][3]/image.size[1], 3)
        lable = (x,y,w,h)
        lable = torch.FloatTensor(lable)

        #print(box)

        patch = image.crop(box)

        if self.transform:
            image = self.transform(image)
            patch = self.transform(patch)

        sample = {'image': image, 'patch': patch, 'coordinate':coordinate, 'lable':lable}

        return sample

"""
data_root = "/home/icv/PyTorch/TrakingData/otb100"
path_dir = os.listdir(data_root)
transform = transforms.Compose([
                transforms.Scale((224,224)),
                transforms.ToTensor()
])

for sample in TrakingDataSet("/home/icv/PyTorch/TrakingData/otb100/Human5", transform):
    print(sample['coordinate'])
    print("<------>")
    print(sample['lable'])

fig = plt.figure()

i=0
for sample in trakingSet:
    print(sample['image'].size)
    print(sample['coordinate'])

    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['patch'])
    i = i+1
    if i == 3:
        plt.show()
        break
"""
