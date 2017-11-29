#!/usr/bin/env python3
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

input = Variable(torch.zeros(200704, 32).cuda())

running_mean = torch.zeros(32).cuda()
running_var = torch.ones(32).cuda()

weight = Parameter(torch.ones(32).cuda())
bias = Parameter(torch.ones(32).cuda())

F.batch_norm(input, running_mean, running_var, weight, bias,
             training=True)

