import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import skimage.io as io
import skimage.color as color

class WhalesSegmentation(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, data_path, image, label, split, drop_last):
        super(WhalesSegmentation, self).__init__()
        'Initialization'
        self.args=args
        self.data_path = data_path
        self.image = image        #lista de la carpeta de las imagenes de cada particion
        self.label = label      #diccionario de imagenes y anotacion (train,val o test)
        #self.resize = transforms.Resize(size=(224,224))
        self.split = split
        self.drop_last=drop_last

        #self.void_classes = [2]
        self.valid_classes = [0, 1]
        self.class_names = ['whale','background']

        #self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        im = self.image[index]
        im = io.imread(self.data_path + im)
        im=Image.fromarray(im)
        #im = self.resize(im)

        tar = self.label[index]
        tar = io.imread(self.data_path + tar)
        #print(tar)
        #print(tar.shape)
        tar = color.rgb2gray(tar)
        #print(tar)
        #print(tar.shape)
        # tar[tar>0.25]=1
        # tar[tar<=0.25]=0
        tar[tar>200]=255
        tar[tar<=200]=0
        #print(tar)
        #print(tar.shape)
        #tar=Image.fromarray(tar)
        #targ= transforms.functional.to_grayscale(tar)
        #tar=np.asarray(targ)
        #tar=tar[..., np.newaxis]
        #print(tar.shape)
        #targ=targ.unsqueeze(-1)
        tar=Image.fromarray(tar,mode='L').convert('1')
        #print(tar)
        #tar = self.resize(tar)
        #print(tar.size)

        sample = {'image': im, 'label': tar}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        #for _voidc in self.void_classes:
            #mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
     from dataloaders.utils import decode_segmap
     from torch.utils.data import DataLoader
     import matplotlib.pyplot as plt
     import argparse

     parser = argparse.ArgumentParser()
     args = parser.parse_args()
     args.base_size = 513
     args.crop_size = 513

     whales_train = WhalesSegmentation(args)

     dataloader = DataLoader(whales_train, batch_size=4, shuffle=True, num_workers=2)

#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["image"].size()[0]):
#             img = sample['image'].numpy()
#             gt = sample['label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='whales')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(211)
#             plt.imshow(img_tmp)
#             plt.subplot(212)
#             plt.imshow(segmap)

#         if ii == 1:
#             break

#     plt.show(block=True)
