import numpy as np
import torch
from utils.DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.reconNet import ReconNet
import torch.optim as optim
import torch.optim.lr_scheduler as s


N, C = 5, 4
loss = nn.NLLLoss2d()
# input is of size N x C x height x width
data = Variable(torch.randn(N, 16, 10, 10))
m = nn.Conv2d(16, C, (3, 3))
# each element in target has to have 0 <= value < C
target = Variable(torch.LongTensor(N, 8, 8).random_(0, C))

print(m(data).size(),target.size())
output = loss(m(data), target)
output.backward()
