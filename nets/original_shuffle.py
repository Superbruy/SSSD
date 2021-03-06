# -*- coding=utf-8 -*-
'''
# @filename  : original_shuffle.py
# @author    : cjr
# @date      : 2021-4-14
# @brief     : shuffle net with fc layer
'''
import torch
import torch.nn as nn
from ss_backbone import BasicConv2d, ShuffleNetUnit


class Shuffle_fc(nn.Module):

    def __init__(self, num_blocks, groups=4, num_classes=10):
        super(Shuffle_fc, self).__init__()

        out_channels = []
        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv_head = BasicConv2d(3, out_channels[0], kernel=3, stride=2, padding=1)
        # self.conv_head = nn.Conv2d(3, out_channels[0], kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            stride=2,
            num_block=num_blocks[0],
            out_channels=out_channels[1],
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            stride=2,
            num_block=num_blocks[1],
            out_channels=out_channels[2],
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            stride=2,
            num_block=num_blocks[2],
            out_channels=out_channels[3],
            groups=groups
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):

        x = self.conv_head(x)  # /2
        x = self.max_pool(x)  # /2
        x = self.stage2(x)  # /2
        x = self.stage3(x)  # /2
        x = self.stage4(x)  # /2
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, stride, num_block, out_channels, groups):
        '''
        each stage contains several shuffle_blocks
        :param block: type of block, default = shufflenet_block
        :param stride: stride in each stage
        :param num_block: how many blocks in each stage
        :param out_channels:
        :param groups:

        :return:
        '''

        # stride parameter for shuffle_block, namely:[2, 1, 1, 1] or [2, 1, 1, 1, 1, 1, 1, 1]
        strides = [stride] + [1] * (num_block - 1)
        container = []
        for s in strides:
            container.append(
                block(
                    s,
                    self.in_channels,
                    out_channels,
                    groups
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*container)


if __name__ == '__main__':
    # s = Shuffle_fc([4, 8, 4])
    # a = torch.rand(2, 3, 224, 224)
    # b = s(a)
    # print(b.shape)

    # torch.save(s.state_dict, 'fc.pth')

    state = torch.load('resnet18-5c106cde.pth')
    state1 = state.popitem(last=True)
    state2 = state.popitem(last=True)

    # print(state)
    for i in state:
        print(i)