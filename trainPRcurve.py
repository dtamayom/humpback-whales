#! /usr/bin/env python
#'Proyecto Ballenas jorobadas BCV train.py'
import time
import argparse
import os.path as osp
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
#from architecture import Net
from Dataloader import DatasetJorobadas
import torch.nn as nn
import numpy as np
import csv
import os
import sklearn.metrics as metrics
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

#argumentos
parser = argparse.ArgumentParser(description='CNN Whales')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='augmented2.pt',
                    help='file on which to save model weights')

numwhales = 66

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

with open('train_final.csv', mode='r') as file1:
    reader = csv.reader(file1)
    train = {rows[0]:rows[1] for rows in reader}
train_im=os.listdir('../data/HumpbackWhales/train_final/')

with open('val_final.csv', mode='r') as file2:
    reader = csv.reader(file2)
    val = {rows[0]:rows[1] for rows in reader}
val_im=os.listdir('../data/HumpbackWhales/val_final/')

with open('test_final.csv', mode='r') as file3:
    reader = csv.reader(file3)
    test = {rows[0]:rows[1] for rows in reader}
test_im=os.listdir('../data/HumpbackWhales/test_final/')

path= '../data/HumpbackWhales/'

##DATA AUGMENTATION
# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=226, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ]),
}

train_loader = torch.utils.data.DataLoader(DatasetJorobadas(train_im, train, '../data/HumpbackWhales/train_final/',
                  image_transforms['train']),batch_size=args.batch_size, shuffle=True, **kwargs)


val_loader = torch.utils.data.DataLoader(DatasetJorobadas(val_im, val, '../data/HumpbackWhales/val_final/', 
                  image_transforms['val']),batch_size=args.batch_size, shuffle=True, **kwargs)

#model = models.vgg16(pretrained=True)
model = models.densenet161(pretrained=False)

for param in model.parameters():
    param.requires_grad = False

n_inputs =model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(n_inputs,numwhales),
                         nn.ReLU(),
                         nn.LogSoftmax(dim=1))


# model.classifier = nn.Sequential(nn.Linear(25088,4096, bias=True),
#                          nn.ReLU(inplace=True),
#                          nn.Dropout(p=0.5, inplace=False),
#                          nn.Linear(4096,4096, bias=True),
#                          nn.ReLU(inplace=True),
#                          nn.Dropout(p=0.5, inplace=False),
#                          nn.Linear(in_features=4096, out_features=numwhales, bias=True),
#                          nn.LogSoftmax(dim=1))

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

optimizer = optim.Adam(model.parameters())
#optimizer = optim.Adam(model.parameters(), lr= 3e-4, betas=(0.9, 0.99), weight_decay=0.0002)
criterion= nn.NLLLoss()

def Train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target=list(map(int,target))
        target=torch.tensor(target)
        target=target.long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float()), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item() * data.size(0)


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def curvita(recall, precision, average_precision, batchnum, epoch):
    m = True
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','goldenrod','darkslategrey','sandybrown','mediumseagreen','sienna'])
    plt.figure(figsize=(15, 15))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    # lines.append(l)
    # labels.append('iso-f1 curves')
    # l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    # lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #             ''.format(average_precision["micro"]))

    for i, color in zip(range(batchnum), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for batch {0} (area = {1:0.2f})'
                    ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(os.path.join('curvas/', 'ep' + str(epoch) + '.jpg'))
    return m

def Validation(epoch):
    with torch.no_grad():
        val_loss=0.0
        val_acc=0.0
        prec=[]
        reca=[]
        fbeta=[] 
        precisionall = dict()
        recallall = dict()
        average_precisionall = dict()

        # Set to evaluation mode
        model.eval()
        # Validation loop
        numbatch=-1
        for data, target in val_loader:
            numbatch+=1
            target=list(map(int,target))
            target=torch.tensor(target)
            target=target.long()

            precision = dict()
            recall = dict()
            average_precision = dict()

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # Forward pass
            output = model(data.float())
            sig=nn.Sigmoid()
            proba = sig(output)
            # Validation loss
            loss = criterion(output.float(), target)
            # Multiply average loss times the number of examples in batch
            val_loss += loss.item() * data.size(0)
            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            pred = pred.cuda()
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            val_acc += accuracy.item() * data.size(0)
            #Calculate metrics
            pre, rec, f, _= metrics.precision_recall_fscore_support(target.cpu().numpy(), pred.cpu().numpy(), average='weighted', zero_division=0)
            prec.append(pre)
            reca.append(rec)
            fbeta.append(f)

            # tar = []
            # for t in range(len(target)):
            #     tar.append([])
            #     for i in range(0, 66):
            #         if target[t] == i: 
            #             tar[t].append(1)
            #         else:
            #             tar[t].append(0) 
            batch_classes = sorted(set(target.cpu().numpy()))
            tar = label_binarize(target.cpu(),classes=[*batch_classes])
                #binary_pred= label_binarize(pred.cpu(),classes=[*range(batch_classes)])
            
            pru = []
            tarcompleto=[]
            b=-1
            for i in batch_classes:
                #precision, recall, fbeta, support= metrics.precision_recall_fscore_support(target, pred.cpu().numpy(), average='binary', pos_label=i)
                b+=1

                precision[i], recall[i], _ = precision_recall_curve(tar[:, b], proba[:, i].cpu())
                pru.append(proba[:, i].cpu().numpy())
                tarcompleto.append(tar[:, b])
                
                average_precision[i] = average_precision_score(tar[:, b], proba[:, i].cpu())

            # A "micro-average": quantifying score on all classes jointly
            precisionall[numbatch], recallall[numbatch], _ = precision_recall_curve(np.array(tarcompleto).ravel(), np.array(pru).ravel())
            average_precisionall[numbatch] = average_precision_score(tarcompleto, pru, average="micro")

            #m = curvita(recall, precision, average_precision,batch_classes)
            
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precisionall[numbatch]))
        if not os.path.exists("curvas"):
            os.mkdir("curvas")
        m = curvita(recallall, precisionall, average_precisionall, numbatch, epoch)
        
        # Calculate average losses
        val_loss = val_loss / len(val_loader.dataset)

        # Calculate average accuracy
        val_acc = val_acc / len(val_loader.dataset)

        val_precision = sum(prec)/len(prec)
        val_recall= sum(reca)/len(reca)
        val_fbeta= sum(fbeta)/len(fbeta)

        print(f'\nEpoch: {epoch} \tValidation Loss: {val_loss:.4f}')
        print(f'\t\t Validation Accuracy: {100 * val_acc:.2f}%')
        print(f'Validation avg Precision: {100*val_precision:.2f}%')
        print(f'Validation avg Recall: {100*val_recall:.2f}%')
        print(f'Validation avg Fbeta: {100*val_fbeta:.2f}%')

        # precision = dict()
        # recall = dict()
        # average_precision = dict()
        # # tar = []
        # # for t in range(len(target)):
        # #     tar.append([])
        # #     for i in range(0, 66):
        # #         if target[t] == i: 
        # #             tar[t].append(1)
        # #         else:
        # #             tar[t].append(0) 
        # batch_classes = sorted(target.cpu().numpy())
        # tar = label_binarize(target.cpu(),classes=[*range(len(batch_classes))])
        #     #binary_pred= label_binarize(pred.cpu(),classes=[*range(batch_classes)])
            
        # pru = []
        # for i in range(len(batch_classes)):
        #     #precision, recall, fbeta, support= metrics.precision_recall_fscore_support(target, pred.cpu().numpy(), average='binary', pos_label=i)
        #     precision[i], recall[i], _ = precision_recall_curve(tar[:, i], proba[:, batch_classes[i]].cpu())
        #     pru.append(proba[:, batch_classes[i]].cpu().numpy())
        #     average_precision[i] = average_precision_score(tar[:, i], proba[:, batch_classes[i]].cpu())

        # for i in range(len(target)):
        #    precision[i], recall[i], _ = precision_recall_curve(tar[i], proba[i])
        #    average_precision[i] = average_precision_score(tar[i], proba[i])

    return(val_loss)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = Validation(0)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            Train(epoch)
            val_loss = Validation(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)
            else:
                adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

