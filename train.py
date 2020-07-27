#! /usr/bin/env python
#'Proyecto Ballenas jorobadas BCV train.py'
import time
import argparse
import os.path as osp
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
#from architecture import Net
from Dataloader import DatasetJorobadas
import torch.nn as nn
import numpy as np
import csv
import os

#argumentos
parser = argparse.ArgumentParser(description='CNN Whales')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

with open('train_final.csv', mode='r') as file1:
    reader = csv.reader(file1)
    train = {rows[0]:rows[1] for rows in reader}
train_im=os.listdir('../data/HumpbackWhales/train_final')

with open('val_final.csv', mode='r') as file2:
    reader = csv.reader(file2)
    val = {rows[0]:rows[1] for rows in reader}
val_im=os.listdir('../data/HumpbackWhales/val_final')

with open('test_final.csv', mode='r') as file3:
    reader = csv.reader(file3)
    test = {rows[0]:rows[1] for rows in reader}
test_im=os.listdir('../data/HumpbackWhales/test_final')

path= '../data/HumpbackWhales/'

train_loader = torch.utils.data.DataLoader(DatasetJorobadas(train_im, train, '../data/HumpbackWhales/train_final'), 
                  batch_size=args.batch_size, shuffle=True, **kwargs)


val_loader = torch.utils.data.DataLoader(DatasetJorobadas(val_im, val, '../data/HumpbackWhales/val_final'), 
                  batch_size=args.batch_size, shuffle=True, **kwargs)

model = Net()

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def Train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def Validation(epoch):
    model.eval()
    val_loss = 0
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        # get the index of the max log-probability
        val_loss += F.nll_loss(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        val_loss = val_loss
        # loss function already averages over batch size
        val_loss /= len(val_loader)
        acccuracy = 100. * correct / len(val_loader.dataset)
        print('\nval set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, len(val_loader.dataset), acccuracy))
    return val_loss


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