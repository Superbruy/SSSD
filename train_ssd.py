# -*- coding=utf-8 -*-
'''
# @filename  : train_ssd.py
# @author    : cjr
# @date      : 2021-4-11
# @brief     : modified train.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import argparse
import os
import time

from nets.shuffle_ssd import build_ssd_s
from PriorBox import voc
from dataset import VOCDetection
from augmentation import SSDAugmentation
from config import detection_collate, xavier, weights_init, MEANS
from loss_fn import MultiBoxLoss

import visdom

BaseDir = os.path.abspath(os.path.dirname(__file__))
VOC_ROOT = os.path.join(BaseDir, "..", "VOC2007")
Weight_Root = os.path.join(BaseDir, "..", "weight")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='SSD training with pytorch')
parser.add_argument('--dataset', default='VOC', type=str)
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')

parser.add_argument('--epoch', default=20, type=int,
                    help='total training epoch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')

parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--save_folder', default=Weight_Root,
                    help='Directory for saving checkpoint models')

parser.add_argument('--resume', default=None, type=str,
                    help='whole net pretrained weights')
parser.add_argument('--basenet', default='shuffle_default.pth',
                    help='base net pretrained weights')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train_ssd():
    # 程序开头将该GPU flag设为true，在卷积参数不变的时候可以加速
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    #  dataset configuration
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))

    # network configuration
    ssd_net = build_ssd_s('train', cfg['min_dim'], cfg['num_classes'])

    # load pretrained weights if possible
    if args.resume:
        print('Loading whole network...')
        ssd_net.load_state_dict(args.resume)
    else:
        # base_weights: my shuffle net with out fc layer
        base_weights = torch.load(args.save_folder + args.basenet)
        base_fcreduced1 = base_weights.popitem(last=True)
        base_fcreduced2 = base_weights.popitem(last=True)
        print('Loading base network...')
        ssd_net.base.load_state_dict(base_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    net = ssd_net
    # put net to gpu
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

    # optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()

    epoch_size = len(dataset) // args.batch_size
    print("An epoch contains {} iterations".format(epoch_size))
    print('Training on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0  # 用于调整学习率
    total_iter = 0  # 总共的迭代次数

    # dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=detection_collate)  # collate_fn:make your own data style in a batch

    for epoch in range(args.epoch):
        print("Epoch: [{}/{}]".format(epoch, args.epoch))
        # loss counters
        conf_loss = 0.
        loc_loss = 0.
        for i, (image, targets) in enumerate(dataloader):
            if args.cuda:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                image, targets = image.to(device), targets.to(device)
            # forward
            t0 = time.time()
            output = net(image)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(output, targets)
            loss = loss_c + loss_l
            loss.backwards()
            optimizer.step()
            t1 = time.time()

            loc_loss += loss_l
            conf_loss += loss_c

            if i % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(i) + ' || Loss: %.4f ||' % (loss.data), end=' ')
            if args.visdom:
                pass
        print(i)
        # update visdom for each epoch
        if epoch % 1 == 0:
            pass
        total_iter += epoch_size
        if total_iter in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        # save checkpoint every 5000 iterations
        if total_iter != 0 and total_iter % 5000 == 0:
            print("Saving state, iter_num:{}".format(total_iter))
            torch.save(net.state_dict(), "weights/s_ssd300" +
                       repr(total_iter) + '.pth')
    torch.save(net.state_dict(), args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    if args.visdom:
        viz = visdom.Visdom()
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    if args.visdom:
        viz = visdom.Visdom()
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )
if __name__ == '__main__':
    train_ssd()