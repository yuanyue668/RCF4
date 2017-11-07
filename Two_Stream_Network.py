import torch
import numpy
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, models, transforms

model = models.vgg16(pretrained=True)

class Two_Stream_Network_v2(nn.Module):
    def __init__(self):
        super(Two_Stream_Network_v2, self).__init__()
        self.net1 = nn.Sequential(
            *list(model.features.children())
        )
        #self.net2 = nn.Sequential(
        #    *list(model.features.children())
        #)
        self.fc1 = nn.Linear(25088,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.lstm = nn.LSTM(
            input_size = 68,
            hidden_size = 4,
            num_layers = 1,
            batch_first = True
        )

    def forward(self, input1, input2, hstate, coordinate):
        output1 = self.net1(input1)
        #output2 = self.net2(input2)

        output1 = output1.view(output1.size(0),-1)          #convolutional layer convert to Linear layer need change the shaper
        #output2 = output2.view(output2.size(0), -1)

        #fc_input = torch.cat((output1,output2),1)
        fc_input = output1
        #print(fc_input)
        fc1_output = self.fc1(fc_input)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        fc4_output = self.fc4(fc3_output)

        lstm_input = torch.cat((fc4_output,coordinate),1)
        #print(lstm_input.size())
        lstm_input = lstm_input.view(1,-1,68)
        #print(lstm_input.size())
        #lstm_input = coordinate
        lstm_output,hstate =self.lstm(lstm_input,hstate)

        return lstm_output,hstate
        #return F.log_softmax(x)
