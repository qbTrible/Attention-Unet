import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import os
import Attention_UNet
import MKDataset
from MyLoss import BCEFocalLoss

path = 'pic_train'
module = 'model/module.pkl'
epochs = 0

net = Attention_UNet.MainNet(16).cuda()
optimizer = torch.optim.Adam(net.parameters())
# optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
# loss_func = BCEFocalLoss()
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=1, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print('module is loaded !')

while True:
    # scheduler.step()
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        xs_ = net(xs)

        loss = loss_func(xs_, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('epochs: {}  batchs: {}  loss: {}'.format(epochs, i, loss))
            torch.save(net.state_dict(), module)
        del xs, xs_, ys, i, loss
    epochs += 1