import torch
import numpy
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()
        self.lstm = nn.LSTM(
            input_size = 16*5*5*2,
            hidden_size = 64,
            num_layer = 1,
            batch_first = True
        )
        self.fc = nn.Linear(64,4)

    def forward(self,x,h_state):
        output, h_state = self.lstm(x,h_state)
        out = self.fc(output[:,-1,:])
        return out, h_state

