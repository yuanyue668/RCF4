import torch
import numpy
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, models, transforms


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

        print(net1_output)

        fc_input =torch.cat((net1_output,net2_output),1)

        print(fc_input)

        fc_output = self.fc1(fc_input)        #what kind of structure of the full-connection layer is?of course ,it is tensor
        final_output = self.fc2(fc_output)

        return F.log_softmax(final_output)



Epoch = 60

cuda = torch.cuda.is_available()

two_stream = Two_Stream_Network()

if cuda:
    print("GPU")
    two_stream = two_stream.cuda();

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

optimizer = optim.SGD(two_stream.parameters(), lr=0.02, momentum=0.75)

def train(epoch):
    two_stream.train()
    for batch_idx,(data,lable) in enumerate(train_loader):
        if cuda:
            data,lable = data.cuda(),lable.cuda()
        data,lable =  Variable(data),Variable(lable)
        optimizer.zero_grad()
        output = two_stream(data,data)
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
    two_stream.eval()
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = two_stream(data,data)

        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()


    test_loss /= len(test_loader)
    print('\nTest set: Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                               100.* correct / len(test_loader.dataset)))


for epoch in range(Epoch):
    train(epoch)
    test(epoch)