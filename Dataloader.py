#Dataloader
import os
import pandas  as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv

def Correction():
    
    return 1


# Dataset class
class DatasetJorobadas(Dataset):
  'Caracteriza dataset para PyTorch'
  def __init__(self, image_IDs, attributes, data_path, transform=transforms.ToTensor()):
        super(DatasetJorobadas, self).__init__()
        'Initialization'
        self.image_IDs = image_IDs        #lista de las imagenes de cada particion
        self.attributes = attributes      #diccionario de imagenes y su lista de anotaciones (train,val o test)
        self.data_path = data_path        #'../data/CelebA_HQ'
        self.transform = transform        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.image_IDs[index]
        # Load data and get attributes
        attribute = self.attributes[ID]
        image = io.imread(self.data_path + '/celeba-256/' + ID)
        if self.transform:
            image = self.transform(image)
        return image, attribute