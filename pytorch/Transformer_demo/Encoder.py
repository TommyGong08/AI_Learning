import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import torch.functional as F
import matplotlib as plt
import seaborn
import Layer as ly
seaborn.set_context(context="talk")  # seaborn风格设置


class Encoder(nn.Module):
    """
    核心编码器是N层的堆栈s
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = ly.clones(layer, N)
        self.norm = ly.LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        让x一次通过encoder每一层
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    构造整个encoder，包括attention模块和feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = ly.clones(ly.SublayerConnection(size, dropout), 2)  # 构造两个残差结构
        self.size = size

    def forward(self, x, mask):
        """
        encoder具体的结构可以查看图中的encoder
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # x和attention模块的结果做残差和
        return self.sublayer[1](x, self.feed_forward)  # x和feed_farword做残差和

