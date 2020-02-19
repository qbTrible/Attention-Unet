# -*- coding: utf-8 -*-
# @Time: 2020/1/23 22:04
# Author: Trible

import torchvision
import UNet
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])
module = 'model/module03.pkl'
path = "pic_test"

img_list = os.listdir(path)
net = UNet.MainNet(16).cuda()
if os.path.exists(module):
    net.load_state_dict(torch.load(module))
net.eval()
# img_name = random.choice(img_list)
for img_name in img_list:
    img = Image.open(os.path.join(path, img_name))
    img = img.resize((512, 512), 1)
    data = transform(img).unsqueeze(0).cuda()
    out_img = net(data).squeeze(0)
    img_save = transforms.ToPILImage()(out_img.cpu())
    img_save.save("predict/"+img_name)
    del img, data, out_img, img_save
    print("%s保持完毕"%img_name)
