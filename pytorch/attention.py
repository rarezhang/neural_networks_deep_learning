#!/usr/bin/python3
# coding=utf-8
"""
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math, copy, time
import matplotlib.pyplot as plt 
import seaborn 
seaborn.set_context(context='talk')
