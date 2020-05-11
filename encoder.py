#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:15:11 2020

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo import get_model

class ResNetBackbone(HybridBlock):
    """ResNet module backbone outputs
    """
    def __init__(self, backbone="resnet34_v2", pretrained_base=True, **kwargs):
        """
        Args:
            backbone: string, backbone name from `gluoncv.model_zoo`.
            pretrained_base: bool, whether to load pretrained model
        """
        super(ResNetBackbone, self).__init__()

        with self.name_scope():
            model = get_model(backbone, pretrained=pretrained_base, **kwargs)

            self.base_conv = model.features[0:5]
            self.block1 = model.features[5:6]
            self.block2 = model.features[6:7]
            self.block3 = model.features[7:8]
            self.block4 = model.features[8:9]
            
    def hybrid_forward(self, F, x):
        base = self.base_conv(x)
        c1 = self.block1(base)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        return c1, c2, c3, c4


if __name__ == "__main__":
    """Unit Test"""
    from mxnet import nd
    x = nd.ones((32, 3, 224, 224))
    model = ResNetBackbone(backbone="resnet34_v2")
    model.initialize(mx.init.MSRAPrelu())

    res = model(x)
    for i in res:
        print(i.shape)