import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])

class MKDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.pic_name = os.listdir(self.path)
        self.mask_name = os.listdir('heatmap02')

    def __len__(self):
        return len(self.pic_name)

    def __getitem__(self, index):
        pic_path = self.path
        mask_path = 'heatmap02'
        pic = Image.open(os.path.join(pic_path, self.pic_name[index]))
        pic = pic.resize((512, 512), 1)
        mask = Image.open(os.path.join(mask_path, self.mask_name[index]))
        mask = mask.resize((512, 512), 1)

        return transform(pic), transform(mask)


if __name__ == '__main__':
    dataset = MKDataset(r'C:\Users\Trible\Desktop\螺钉')
    print(dataset[0])
