from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, whales
from torch.utils.data import DataLoader
import os

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'whales':
        train_path = "../../data/HumpbackWhales/segmentacion/train/"
        val_path = "../../data/HumpbackWhales/segmentacion/val/"
        test_path = "../../data/HumpbackWhales/segmentacion/test/"

        train_set = whales.WhalesSegmentation(args, data_path=train_path+'mask_images_train/', image=os.listdir(train_path+'mask_images_train/'),label=os.listdir(train_path+'mask_train/'),split='train', drop_last=True)
        val_set = whales.WhalesSegmentation(args, data_path=val_path+'mask_images_val/', image=os.listdir(val_path+'mask_images_val/'),label=os.listdir(val_path+'mask_val/'),split='val', drop_last=True)
        test_set = whales.WhalesSegmentation(args, data_path=test_path, image=os.listdir(test_path),label= None, split='test', drop_last=True) #,label=os.listdir(test_path)
        
        num_class=train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader,val_loader,test_loader,num_class

    else:
        raise NotImplementedError

