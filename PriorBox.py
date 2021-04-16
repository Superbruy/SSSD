# -*- coding=utf-8 -*-
'''
# @filename  : PriorBox.py
# @author    : cjr
# @date      : 2021-4-6
# @brief     : generate prior box
'''
import torch
from math import sqrt
from itertools import product

voc = {
    #'num_classes': 21,
    #'lr_steps': (80000, 100000, 120000),
    #'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    # 由配置文件而来的一些配置信息
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # 生成所有的priorbox需要相应特征图的信息
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):  # 'feature_maps': [38, 19, 10, 5, 3, 1],
            for i, j in product(range(f), repeat=2):  # (0,0),(0,1),(0,2),(0,range(f)),...
                # f_k 为每个特征图上的格点 对应原图的大小，相当于感受野
                f_k = self.image_size / self.steps[k]  # self.image_size=300 'steps': [8, 16, 32, 64, 100, 300]
                # 求每个box的中心坐标  将中心点坐标转化为 相对于 特征图的 相对坐标 （0，1）
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 对应{Sk,Sk}大小的priorbox
                s_k = self.min_sizes[k] / self.image_size  # 'min_sizes': [30, 60, 111, 162, 213, 264],
                mean += [cx, cy, s_k, s_k]
                # 对应{sqrt(Sk*Sk+1), sqrt(Sk*Sk+1)}大小的priorbox
                s_k_prime = sqrt(
                    s_k * (self.max_sizes[k] / self.image_size))  # 'max_sizes': [60, 111, 162, 213, 264, 315]
                mean += [cx, cy, s_k_prime, s_k_prime]
                # 对应比例为2、 1/2、 3、 1/3的priorbox
                for ar in self.aspect_ratios[k]:  # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # 将所有的priorbox汇集在一起
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

if __name__ == '__main__':
    priorbox = PriorBox(voc)  # 实例化一个对象 之后 才可调用对象里的输出
    output = priorbox.forward()
    print(output.shape)