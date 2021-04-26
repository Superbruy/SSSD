# -*- coding=utf-8 -*-
'''
# @filename  : shuffle_ssd.py
# @author    : cjr
# @date      : 2021-4-5
# @brief     : shuffle + ssd
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

import os

from nets.ss_backbone import Shuffle_ssd
from PriorBox import PriorBox, voc
from detection import Detect

# construct extra layers
# this part can refer to https://github.com/amdegroot/ssd.pytorch

extras = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
mbox = [4, 6, 6, 6, 4, 4]


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to backbone for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def get_heads(cfg, extra_layers, num_classes):
    loc_layers = []
    conf_layer = []

    for k in range(2):
        loc_layers += [nn.Conv2d(272 * (k + 1), cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layer += [nn.Conv2d(272 * (k + 1), cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layer += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layer)


class S_SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(S_SSD, self).__init__()
        self.phase = phase
        if self.phase != "test" and self.phase != "train":
            raise Exception("phase must be train or test, got {} instead".format(self.phase))

        self.size = size
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.priors.requires_grad = False

        self.base = base
        self.extras = extras
        self.num_classes = num_classes

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):

        sources = []
        loc = []
        conf = []

        net_half1 = nn.Sequential(*list(self.base.children())[:-2])
        x1 = net_half1(x)  # [b, 272, 38, 38]
        sources.append(x1)

        net_half2 = nn.Sequential(*list(self.base.children())[:-1])
        x2 = net_half2(x)  # [b, 544, 19, 19]
        sources.append(x2)

        for k, v in enumerate(self.extras):
            x2 = F.relu(v(x2), inplace=True)
            if k % 2 == 1:
                sources.append(x2)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # reshape tensor to [bn, c*h*w+c*h*w+...]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            # map_location=lambda storage, loc: storage
                                            # this parameter is used for multi gpu transfer or cpu gpu transfer
                                            ))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_ssd_s(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        raise Exception("ERROR: Phase: " + phase + " not recognized")

    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_net = Shuffle_ssd([4, 8, 4])
    extra_layers = add_extras(extras, 544)
    head = get_heads(mbox, extra_layers, 21)
    return S_SSD(phase, size, base_net, extra_layers, head, num_classes)


if __name__ == '__main__':
    extra_layers = add_extras(extras, 544)
    print(extra_layers)
    # x = torch.rand(2, 544, 19, 19)
    # for k, v in enumerate(extra_layers):
    #     x = F.relu(v(x), inplace=True)
    #     if k % 2 == 1:  # 间隔一层
    #         print(x.shape)
    head = get_heads(mbox, extra_layers, 21)
    # for i in range(len(head[0])):
    #     print(head[0][i])
    # for i in range(len(head[1])):
    #     print(head[1][i])
    base_net = Shuffle_ssd([4, 8, 4])
    net = S_SSD('train', 300, base_net, extra_layers, head, 21)
    print(net.base)
    # t = torch.rand(2, 3, 300, 300)
    # g = net(t)
    # print(g[2].shape)
