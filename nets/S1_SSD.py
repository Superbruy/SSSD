# -*- coding=utf-8 -*-
'''
# @filename  : S1_SSD.py
# @author    : Superbruy
# @date      : 2021-5-24
# @brief     : shuffle v1 + ssd
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import os

from nets.ss_backbone import Shuffle_ssd
from PriorBox import PriorBox, voc
from detection import Detect, L2Norm

# construct extra layers
# this part can refer to https://github.com/amdegroot/ssd.pytorch

extras = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
mbox = [4, 6, 6, 6, 4, 4]

class S1_SSD(nn.Module):
    """
    changes include:
    def __init__: self.detect
    def forward: if phase == 'test'
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(S1_SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = self.priorbox.forward()
        self.priors.requires_grad = False
        self.size = size

        # S1_SSD network
        # input base is not a list, so pack all layers into a list
        self.s1_bb = nn.ModuleList([*list(base.children())])
        # Layer learns to scale the l2 normalized features from stage 2
        self.L2Norm = L2Norm(272, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        """
        sources = list()
        loc = list()
        conf = list()

        # apply s1_backbone up to stage 2, output shape: [b, 272, 38, 38]
        for k in range(3):
            x = self.s1_bb[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply s1_bb up to stage 3
        # output shape: [b, 544, 19, 19]
        x = self.s1_bb[3](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # if self.phase == "test":
        #     output = self.detect(
        #         loc.view(loc.size(0), -1, 4),                   # loc preds
        #         self.softmax(conf.view(conf.size(0), -1,
        #                      self.num_classes)),                # conf preds
        #         self.priors.type(type(x.data))                  # default boxes
        #     )
        if self.phase == "test":
            output = self.detect.apply(21, 0, 200, 0.01, 0.45,
                                       loc.view(loc.size(0), -1, 4),  # loc preds
                                       self.softmax(conf.view(-1,
                                                              21)),  # conf preds
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
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



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


def build_s1_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        raise Exception("ERROR: Phase: " + phase + " not recognized")

    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_net = Shuffle_ssd([4, 8, 4])
    extra_layers = add_extras(extras, 544)
    head = get_heads(mbox, extra_layers, 21)
    return S1_SSD(phase, size, base_net, extra_layers, head, num_classes)

if __name__ == '__main__':
    extra_layers = add_extras(extras, 544)
    head = get_heads(mbox, extra_layers, 21)
    base_net = Shuffle_ssd([4, 8, 4])
    net = S1_SSD('train', 300, base_net, extra_layers, head, 21)
    # print(net.extras)
    t = torch.rand(2, 3, 300, 300)
    g = net(t)
    print(g[2].shape)