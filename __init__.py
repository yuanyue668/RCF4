import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from TrakingDataSet import TrakingDataSet

from Two_Stream_Network import *

cuda = torch.cuda.is_available()

two_stream = Two_Stream_Network_v2()
#one_stream = One_Stream_Network()

#def get_features_hook(self, input, output):
    #print "Feature:"
    #print(output.data.cpu().numpy())
    #grid = utils.make_grid(features)
    #plt.imshow(grid.numpy().transpose((1,2,0)))

#two_stream.net1[2].register_forward_hook(get_features_hook)

Epoch = 20000
batch_size = 1


if cuda:
    print("GPU")
    two_stream = two_stream.cuda()

#optimizer = torch.optim.Adam(two_stream.parameters(),lr = 0.01)
optimizer = torch.optim.SGD(two_stream.parameters(), lr=0.1)
#optimizer = optim.SGD(two_stream.parameters(), lr=0.02, momentum=0.75)

loss_func = nn.MSELoss()
data_root = "/home/icv/PyTorch/TrakingData/otb100"
path_dir = os.listdir(data_root)
transform = transforms.Compose([
                transforms.Scale((224,224)),
                transforms.ToTensor()
])


def traking_train(epoch):
    two_stream.train()
    hstate = None
    for path in path_dir:
        data_path = os.path.join(data_root, path)
        print(data_path)
        trakingset = TrakingDataSet(data_path, transform)
        dataloader = torch.utils.data.DataLoader(trakingset, batch_size = batch_size, shuffle=False)
        for batch_indx,sample in enumerate(dataloader):
            if cuda:
                image,lable,coordinate,patch = sample['image'].cuda(),sample['lable'].cuda(),sample['coordinate'].cuda(),sample['patch'].cuda()
            #coordinate = coordinate.view(1,-1,4)
            image,lable,coordinate,patch =  Variable(image),Variable(lable),Variable(coordinate),Variable(patch)

            #output = two_stream_v2(data,data)
            #print("ImageBatch:{}".format(len(image)))
            output,hstate = two_stream(image,patch,hstate,coordinate)
            h = Variable(hstate[0].data)  # repack the hidden state, break the connection from last iteration
            c = Variable(hstate[1].data)
            hstate = (h, c)
            #output = one_stream(data)
            loss = loss_func(output,lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_indx % 10 == 0:
                prediction = output.data.cpu().numpy()
                groudth = lable.data.cpu().numpy()
                #for i in range(batch_indx):
                print ('Train Epoch: [{}/{}] Prediction:{} Grounth:{} Loss: {:.6f} '.format(
                        epoch, Epoch,prediction[0][0], groudth[0], loss.data[0]))

for epoch in range(Epoch):
    traking_train(epoch)