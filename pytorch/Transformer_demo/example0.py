import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import torch.functional as F
import matplotlib as plt
import seaborn
import Layer as ly
import train as tr
import Model as md


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False).long()
        tgt = Variable(data, requires_grad=False).long()
        yield tr.Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


# Train the simple copy task.
V = 11
criterion = tr.LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = md.make_model(V, V, N=2)
model_opt = tr.NoamOpt(model.src_embed[0].d_model, 1, 400, \
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    tr.run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(tr.run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))