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

class ConvBlock(HybridBlock):
    """ Base Convolution Block
    """
    def __init__(self, output_channels, kernel_size, 
                       padding=0, 
                       activation='relu', 
                       norm_layer=nn.BatchNorm):
        """
        Args:
            output_channels: int, the number of Convolution operator kernel.
            kernel_size: int or tuple, kernel size of convolution operator. e.g. 3 or (3, 3)
            padding: int, the number of padding for feature map while convolution.
            activation: str, the name of activation operator.
            norm_layer: nn.module, maybe we could use other normalizatiion layers.
                        e.g. nn.LayerNorm, nn.BatchNorm
        """
        
        super().__init__()
        self.body = nn.HybridSequential()
        self.body.add(
            nn.Conv2D(output_channels, 
                      kernel_size=kernel_size,
                      padding=padding,
                      activation=activation),
            norm_layer(in_channels=output_channels)
        )

    def hybrid_forward(self, F, x):
        return self.body(x)

class _DecoderBlock(HybridBlock):
    """ Base Decoder Block.
    """
    def __init__(self, output_channels, norm_layer=nn.BatchNorm):
        """
        Args:
            output_channels: int, the number of Convolution operator kernel.
            norm_layer: nn.module, maybe we could use other normalizatiion layers.
        """
        super(_DecoderBlock, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(ConvBlock(output_channels, 3, padding=1, norm_layer=norm_layer))
            self.block.add(ConvBlock(output_channels, 3, padding=1, norm_layer=norm_layer))
            
        
