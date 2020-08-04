import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class WhalesSegmentation(data.Dataset):
    NUM_CLASSES = 1

    def __init__(self, args, root=Path.db_root_dir('whales'), image, label, split):
        super(WhalesSegmentation, self).__init__()
        'Initialization'
        self.image = image        #lista de la carpeta de las imagenes de cada particion
        self.label = label      #diccionario de imagenes y anotacion (train,val o test)
        self.root = root        #'../data/HumpbackWhales/'
        self.resize = transforms.Resize(size=(224,224))
        self.split = split

        self.void_classes = [0, -1]
        self.valid_classes = [1, 2]
        self.class_names = ['whale','background']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        im = self.image[index]
        im = io.imread(self.data_path + im)
        im=Image.fromarray(im)
        im = self.resize(im)

        tar = self.label[index]
        tar = io.imread(self.data_path + tar)
        tar=Image.fromarray(tar)
        tar = self.resize(tar)

        sample = {'image': im, 'label': tar}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
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

# if __name__ == '__main__':
#     from dataloaders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import argparse

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 513
#     args.crop_size = 513

#     whales_train = WhalesSegmentation(args)

#     dataloader = DataLoader(whales_train, batch_size=2, shuffle=True, num_workers=2)

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
