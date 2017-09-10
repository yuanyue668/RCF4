import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable
from Two_Stream_Network import Two_Stream_Network
from Two_Stream_Network import One_Stream_Network
from Two_Stream_Network import *
from Rnn import Rnn

Epoch = 60

two_stream = Two_Stream_Network()

#one_stream = One_Stream_Network()

#two_stream_v2 = Two_Stream_Network_v2()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('',
                   train = True,
                   download = True,
                   transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = 32,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('',
                   train = False,
                   transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = 16,
    shuffle = True
)

#optimizer = torch.optim.Adam(two_stream_v2.parameters(),lr = 0.01)

optimizer = optim.SGD(two_stream.parameters(), lr=0.02, momentum=0.5)

#loss_func = nn.MSELoss()



def train(epoch):
    for batch_idx,(data,lable) in enumerate(train_loader):
        data,lable =  Variable(data),Variable(lable)
        optimizer.zero_grad()
        #output = two_stream_v2(data,data)
        output = two_stream(data,data)
        #output = one_stream(data)
        loss = F.nll_loss(output,lable)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                     epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        #output = two_stream_v2(data,data)
        output = two_stream(data,data)
        #output = one_stream(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()


    test_loss /= len(test_loader)
    print('\nTest set: Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                               100.* correct / len(test_loader.dataset)))

for epoch in range(Epoch):
    train(epoch)
    test(epoch)
