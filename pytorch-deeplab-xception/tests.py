import argparse
import os
import numpy as np 
import tqdm
import torch
import json

#from Dict
from dataloaders import make_data_loader
from PIL import Image
from modeling.deeplab import *
from dataloaders.utils import get_whales_labels
from dataloaders.__init__ import make_data_loader
from utils.metrics import Evaluator
from torchvision import transforms
from easydict import EasyDict

class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.model):
            raise RuntimeError("no checkpoint found at '{}'".format(args.model))
        self.args = args
        self.color_map = get_whales_labels()
        self.nclass = args.num_class

        #Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)
        
        self.model = model
        device = torch.device('cpu')
        checkpoint = torch.load(args.model, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.evaluator = Evaluator(self.nclass)

    def save_image(self, array, id, op):
        text = 'gt'
        if op == 0:
            text = 'pred'
        file_name = id
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]
    
        rgb = np.dstack((r, g, b))
        save_img = Image.fromarray(rgb.astype('uint8'))
        save_img.save(self.args.save+os.sep+file_name)


    def inference(self):
        self.model.eval()
        self.evaluator.reset()
        
        # 추론할 데이터 경로와 저장할 장소
        DATA_DIR = self.args.data_path
        SAVE_DIR = self.args.save_path
                        
        
        for idx, test_file in enumerate(os.listdir(DATA_DIR)):
            if test_file == '.DS_Store':
                continue
            test_img = Image.open(os.path.join(DATA_DIR, test_file)).convert('RGB')
            test_array = np.array(test_img).astype(np.float32)
            image_id, extension = test_file.split('.')[0], test_file.split('.')[-1]

            # Normalize
            test_array /= 255.0
            test_array -= (0.485, 0.456, 0.406)
            test_array /= (0.229, 0.224, 0.225)
            width = test_array.shape[1]
            height = test_array.shape[0]

            inference_imgs = np.zeros((height, width), dtype=np.float32)
            # count = 0
            for i in range(0, height, self.args.crop_size):
                tmp_i = i
                if i + self.args.crop_size > height:
                    i = height - self.args.crop_size
                for j in range(0, width, self.args.crop_size):
                    tmp_j = j
                    if j + self.args.crop_size > width:
                        j = width - self.args.crop_size
    
                    test_crop_array = test_array[i:i+self.args.crop_size,j:j+self.args.crop_size,:]
                    test_crop_array = test_crop_array.transpose((2, 0, 1))
                    test_crop_array_batch = np.expand_dims(test_crop_array, axis=0)
                    test_crop_tensor = torch.from_numpy(test_crop_array_batch)
                    
                    with torch.no_grad():
                        output = self.model(test_crop_tensor)
                    pred = output.data.cpu().numpy()
                    pred=np.argmax(pred, axis=1)*255
                    #pred = np.round(pred)*255
                    inference_imgs[i:i+self.args.crop_size, j:j+self.args.crop_size] = pred[0][:, :]

            print('inference ... {}/{}'.format(idx+1, len(os.listdir(DATA_DIR))))
            # gray mode
            save_image = Image.fromarray(inference_imgs.astype('uint8'))
            save_image.save(os.path.join(self.args.save_path,image_id+'.'+extension))


def main():
    with open('config.json') as f:
        args = json.load(f)
    args = EasyDict(args['inference'])

    tester = Tester(args)

    if args.inference:
        print('predict...')
        tester.inference()

if __name__ == "__main__":
    main()