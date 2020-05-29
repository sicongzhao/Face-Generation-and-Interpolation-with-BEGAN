#!/usr/bin/env python
# Author: Sicong Zhao

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(batch_size, data_path, input_dim):
    '''
    Load data.

    Parameters:
        batch_size (int): the number of image per batch
        data_path (str): the file path of the input image

    Return:
        DataLoader
    '''
    transform = transforms.Compose([
                    transforms.Resize(input_dim),
                    transforms.CenterCrop(input_dim),
                    transforms.ToTensor()
                ])
    celeba_data = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(celeba_data,batch_size=batch_size,shuffle=True)