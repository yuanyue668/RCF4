import torch
import numpy
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

class One_Stream_Network(nn.Module):
    def __init__(self):
        super(One_Stream_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Two_Stream_Network(nn.Module):
    def __init__(self):
        super(Two_Stream_Network, self).__init__()
        self.net1 = One_Stream_Network()
        self.net2 = One_Stream_Network()
        self.fc1 = nn.Linear(20,120)
        self.fc2 = nn.Linear(120,10)

    def forward(self,net1_input,net2_input):
        net1_output = self.net1(net1_input)
        net2_output = self.net2(net2_input)

        fc_input =torch.cat((net1_output,net2_output),1)

        #print(fc_input)

        fc_output = self.fc1(fc_input)        #what kind of structure of the full-connection layer is?of course ,it is tensor
        final_output = self.fc2(fc_output)

        return F.log_softmax(final_output)

#First test the two stream network work or not

class Two_Stream_Network_v2(nn.Module):
    def __init__(self):
        super(Two_Stream_Network_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.conv3 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv4_drop = nn.Dropout2d()
        self.fc3 = nn.Linear(320, 50)
        self.fc4 = nn.Linear(50, 10)

        self.fc5 = nn.Linear(20, 10)

    def forward(self, input1,input2):
        input1 = F.relu(F.max_pool2d(self.conv1(input1), 2))
        input1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input1)), 2))
        input1 = input1.view(-1, 320)
        input1 = F.relu(self.fc1(input1))
        input1 = F.dropout(input1, training=self.training)
        input1 = self.fc2(input1)

        input2 = F.relu(F.max_pool2d(self.conv3(input2), 2))
        input2 = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(input2)), 2))
        input2 = input2.view(-1, 320)
        input2 = F.relu(self.fc3(input2))
        input2 = F.dropout(input2, training=self.training)
        input2 = self.fc4(input2)


        fc5_input = torch.cat((input1,input2),1)
        fc5_output = self.fc5(fc5_input)

        return F.log_softmax(fc5_output)

        return F.log_softmax(x)