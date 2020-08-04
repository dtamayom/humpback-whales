#! /usr/bin/env python
import os
import cv2

if not os.path.exists("../data/HumpbackWhales/segmentacion"):
    os.mkdir("../data/HumpbackWhales/segmentacion")
if not os.path.exists("../data/HumpbackWhales/segmentacion/train"):
    os.mkdir("../data/HumpbackWhales/segmentacion/train")
if not os.path.exists("../data/HumpbackWhales/segmentacion/test"):
    os.mkdir("../data/HumpbackWhales/segmentacion/test")
if not os.path.exists("../data/HumpbackWhales/segmentacion/val"):
    os.mkdir("../data/HumpbackWhales/segmentacion/val") 

if not os.path.exists("../data/HumpbackWhales/segmentacion/val/mask_val"):
    os.mkdir("../data/HumpbackWhales/segmentacion/val/mask_val")
if not os.path.exists("../data/HumpbackWhales/segmentacion/train/mask_train"):
    os.mkdir("../data/HumpbackWhales/segmentacion/train/mask_train")

if not os.path.exists("../data/HumpbackWhales/segmentacion/val/mask_images_val"):
    os.mkdir("../data/HumpbackWhales/segmentacion/val/mask_images_val")
if not os.path.exists("../data/HumpbackWhales/segmentacion/train/mask_images_train"):
    os.mkdir("../data/HumpbackWhales/segmentacion/train/mask_images_train")

mask_path="../data/HumpbackWhales/segmentacion/masks_final/" #masks in RGB
train_path="../data/HumpbackWhales/train/"
mask_train_path="../data/HumpbackWhales/segmentacion/train/mask_train"
mask_val_path="../data/HumpbackWhales/segmentacion/val/mask_val"
train_final_path="../data/HumpbackWhales/train_final/"

c=1
for mask in os.listdir(mask_path):
    imagen = cv2.imread(mask_path + mask, 1)
    if c <= 445:
        cv2.imwrite(os.path.join("../data/HumpbackWhales/segmentacion/train/mask_train/",mask), imagen)
        print(c)
        c+=1
    else:
        cv2.imwrite(os.path.join("../data/HumpbackWhales/segmentacion/val/mask_val/",mask), imagen)
        print(c)
        c+=1

c=1
for mask in os.listdir(mask_train_path):
    imagen = cv2.imread(train_path + mask, 1)
    cv2.imwrite(os.path.join("../data/HumpbackWhales/segmentacion/train/mask_images_train/",mask), imagen)
    print(c)
    c+=1

c=1
for mask in os.listdir(mask_val_path):
    imagen = cv2.imread(train_path + mask, 1)
    cv2.imwrite(os.path.join("../data/HumpbackWhales/segmentacion/val/mask_images_val/",mask), imagen)
    print(c)
    c+=1

c=1
for im in os.listdir(train_final_path):
    imagen = cv2.imread(train_final_path + im, 1)
    cv2.imwrite(os.path.join("../data/HumpbackWhales/segmentacion/test/",im), imagen)
    print(c)
    c+=1