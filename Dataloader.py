#! /usr/bin/env python
'Proyecto Ballenas jorobadas BCV train.py'
import os
import pandas  as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv

def Correction(path_img, path_label):
    label= pd.read_csv(path_label)
    label=label[label.Id != 'new_whale']
    count=label.groupby('Id').size()
    count=count[count>=10]
    dict_count=count.to_dict()
    dict_train=dict(zip(list(label.Image), list(label.Id)))
    dict_final={}
    for im in dict_train:
          Id=dict_train[im]
          if Id in dict_count:
                dict_final[im]=Id
    dict_train={}
    dict_val={}
    dict_test={}

    return dict_final

# Dataset class
class DatasetJorobadas(Dataset):
  'Caracteriza dataset para PyTorch'
  def __init__(self, image, label, data_path, transform=transforms.ToTensor()):
        super(DatasetJorobadas, self).__init__()
        'Initialization'
        self.image = image        #lista de las imagenes de cada particion
        self.label = label      #diccionario de imagenes y anotacion (train,val o test)
        self.data_path = data_path        #'../data/HumpbackWhales'
        self.transform = transform        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image)

  def __getitem__(self, index)
        'Generates one sample of data'
        ID = self.image[index]
        # Load data and get attributes
        label = self.label[ID]
        image = io.imread(self.data_path + ID)
        if self.transform:
            image = self.transform(image)
        return image, label