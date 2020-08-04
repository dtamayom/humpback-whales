#! /usr/bin/env python
import os
import cv2

if not os.path.exists("../data/HumpbackWhales/mask_images"):
    os.mkdir("../data/HumpbackWhales/mask_images")

mask_path="../data/HumpbackWhales/masks_final"
train_path="../data/HumpbackWhales/train/"

c=1
for mask in os.listdir(mask_path):
    imagen = cv2.imread(train_path + mask, 1)
    cv2.imwrite(os.path.join("../data/HumpbackWhales/mask_images/",mask), imagen)
    print(c)
    c+=1
