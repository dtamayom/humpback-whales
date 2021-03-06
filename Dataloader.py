#! /usr/bin/env python
'Proyecto Ballenas jorobadas BCV train.py'
import os
import cv2
import pandas  as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from PIL import Image

def Correction(path_img, path_label):
    label= pd.read_csv(path_label)
    label=label[label.Id != 'new_whale']
    count=label.groupby('Id').size()
    count=count[count>=20]
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
    dict_guia={}
    #Carpetas
    if not os.path.exists("../data/HumpbackWhales/val_final"):
        os.mkdir("../data/HumpbackWhales/val_final")
    if not os.path.exists("../data/HumpbackWhales/test_final"):
        os.mkdir("../data/HumpbackWhales/test_final")
    if not os.path.exists("../data/HumpbackWhales/train_final"):
        os.mkdir("../data/HumpbackWhales/train_final")
    iter=-1
    for ima in dict_final:
          ids = dict_final[ima]
          if not ids == "listo":
            iter+=1
            print(iter)
            test_con = 0
            val_con = 0
            max_test = dict_count[ids]*0.25
            max_val = dict_count[ids]*0.25
            for im in dict_final:
                  if ids == dict_final[im]:
                        if test_con < max_test:
                              dict_test[im] =iter
                              dict_final[im] = "listo"
                              test_con += 1
                              imagen = cv2.imread(path_img + im, 1)
                              cv2.imwrite("../data/HumpbackWhales/test_final/" + im, imagen)
                        elif val_con < max_val:
                              dict_val[im] =iter
                              dict_final[im] = "listo"
                              val_con += 1
                              imagen = cv2.imread(path_img + im, 1)
                              cv2.imwrite("../data/HumpbackWhales/val_final/" + im, imagen)
                        else:
                              dict_train[im] =iter
                              dict_final[im] = "listo" 
                              imagen = cv2.imread(path_img + im, 1)
                              cv2.imwrite("../data/HumpbackWhales/train_final/" + im, imagen) 
            dict_guia[ids]=iter

    w = csv.writer(open("guia_anotaciones.csv", "w"))
    for ids, num in dict_guia.items():
            w.writerow([ids, num])
    w1 = csv.writer(open("train_final.csv", "w"))
    for img, att in dict_train.items():
            w1.writerow([img, att])
    w2 = csv.writer(open("test_final.csv", "w"))
    for img, att in dict_test.items():
            w2.writerow([img, att])
    w3 = csv.writer(open("val_final.csv", "w"))
    for img, att in dict_val.items():
            w3.writerow([img, att])
    return dict_final, dict_test, dict_val, dict_train

#     f, g, h, j = Correction("../data/HumpbackWhales/train/", "../data/HumpbackWhales/train.csv")
#print('Total: ',str(len(f)),'\n')
#print('En test: ', str(len(g)),' (',str((len(g)/len(f))*100),'%)')
#print('En val: ', str(len(h)),' (',str((len(h)/len(f))*100),'%)')
#print('En train: ', str(len(j)),' (',str((len(j)/len(f))*100),'%)')

# Dataset class
class DatasetJorobadas(Dataset):
  'Caracteriza dataset para PyTorch'
  def __init__(self, image, label, data_path, mascapath, transform):
        super(DatasetJorobadas, self).__init__()
        'Initialization'
        self.image = image        #lista de la carpeta de las imagenes de cada particion
        self.label = label      #diccionario de imagenes y anotacion (train,val o test)
        self.data_path = data_path
        self.mascapath = mascapath        #'../data/HumpbackWhales/'
        self.transform = transform
        self.resize = transforms.Resize(size=(224,224))       

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image) 

  def __getitem__(self, index):
        'Generates one sample of data'
        im = self.image[index]
        # Load data and get attributes
        label = self.label[im]
        image = io.imread(self.data_path + im)
        im_mask = io.imread(self.mascapath + im)
        #image = transform.resize(image,(224,224))
        image=Image.fromarray(image)
        image = self.resize(image)
        im_mask=Image.fromarray(im_mask)
        im_mask = self.resize(im_mask)
        #image=Image.fromarray(image)
        #print(image.shape())
        if self.transform:
            im_mask = self.transform(im_mask)
        if self.transform:
            image = self.transform(image)
        #4 Canal 
        image_input = torch.cat([image, im_mask], dim=0)
        
        return image_input, label