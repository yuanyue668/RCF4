import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
from TrakingDataSet import TrakingDataSet

class x_lstm(nn.Module):
    def __init__(self):
        super(x_lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=4,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x, h_state):
        output, h_state = self.lstm(x, h_state)
        return output, h_state


x_lstm = x_lstm()
if torch.cuda.is_available():
    x_lstm = x_lstm.cuda()

optimizer = torch.optim.SGD(x_lstm.parameters(), lr=0.02)

loss_fun = nn.MSELoss()

data_root = "/home/icv/PyTorch/TrakingData/otb100"
path_dir = os.listdir(data_root)
transform = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor()
])


def train(epoch):

    h_state = None
    for path in path_dir:
        data_path = os.path.join(data_root, path)
        print(data_path)
        trakingset = TrakingDataSet(data_path, transform)
        dataloader = torch.utils.data.DataLoader(trakingset, batch_size=4, shuffle=False)
        for batch_indx, sample in enumerate(dataloader):
            coordinate, lable = sample['coordinate'], sample['lable']
            coordinate = coordinate.view(4, -1, 4)
            if torch.cuda.is_available():
                coordinate, lable = coordinate.cuda(), lable.cuda()
            coordinate, lable = Variable(coordinate), Variable(lable)
            # print(coordinate[0])
            # print(lable[0])
            output, h_state = x_lstm(coordinate, h_state)
            h = Variable(h_state[0].data)  # repack the hidden state, break the connection from last iteration
            c = Variable(h_state[1].data)
            h_state = (h, c)
            loss = loss_fun(output, lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre = output.data.cpu().numpy()
            gt = lable.data.cpu().numpy()
            print("Epoch:{}/{} p:{} g:{} loss:{}".format(epoch,50,pre[0][0],gt[0],loss.data.cpu().numpy()[0]))


for i in range(50):
    train(i)