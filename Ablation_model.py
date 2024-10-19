#1. Import library---Following by a-z.
#1) External

import argparse
import copy
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import random
from sklearn.manifold import TSNE
from sklearn import linear_model
import scipy
from scipy.io import loadmat
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import time

#2)Internal

from basic_structure import *
from basic_process import *
from data_loader import *
from utils import *

class SpatialInterpolation(nn.Module):
    def __init__(self):
        super(SpatialInterpolation, self).__init__()
        self.linear = nn.Linear(1, 1)  # Learnable parameter for spatial interpolation

    def forward(self, x):
        return self.linear(x)


# Define model using MultiheadAttention for attention mechanism
class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionMechanism, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, input_data):
        # Compute attention
        attn_output, _ = self.multihead_attn(input_data, input_data, input_data)
        return attn_output