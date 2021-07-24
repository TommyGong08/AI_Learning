import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import torch.functional as F
import matplotlib as plt
import seaborn
import Layer as ly
import Attention as atten
import Encoder as ec
import Decoder as dc


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    通过超参数构造一个模型
    """
    c = copy.deepcopy
    attn = atten.MultiHeadAttention(h, d_model)
    ff = atten.PositionFeedForward(d_model, d_ff, dropout)
    position = atten.PositionalEncoding(d_model, dropout)
    model = ly.EncoderDecoder(
        ec.Encoder(ec.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        dc.Decoder(dc.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(atten.Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(atten.Embeddings(d_model, tgt_vocab), c(position)),
        ly.Generator(d_model, tgt_vocab))

    for p in model.parameters():
       if p.dim() > 1 :
           nn.init.xavier_uniform(p)
    return model