from __future__ import absolute_import, division, print_function

from datetime import datetime
import os
import torch.nn as nn
import torch
import numpy as np
import random
import sys
import pickle
from functools import reduce
from numpy.lib.scimath import log
from scipy import interpolate

# -----------------------------------------------------------------------------------------------------------#
# General auxilary functions
# -----------------------------------------------------------------------------------------------------------#
def list_mult(L):
    return reduce(lambda x, y: x*y, L)

# -----------------------------------------------------------------------------------------------------------#
def seed_setup(seed, deep_fix=False, block_cudnn=False):
   
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deep_fix:
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False 
        if block_cudnn:
            torch.backends.cudnn.enabled = False 

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#-------------------------------------------added by Bojian-------------------------------------------------------
def get_CDT_params(num_train_samples, gamma, device):
    return torch.tensor((np.array(num_train_samples)/np.max(np.array(num_train_samples)))**gamma, dtype=torch.float32, device=device)


def get_init_dy(device, num_classes=2):
    dy = torch.ones([num_classes], dtype=torch.float32, device=device)
    dy.requires_grad = True
    return dy


def get_init_ly(device, num_classes=2):
    ly = torch.zeros([num_classes],dtype=torch.float32, device=device)
    ly.requires_grad = True
    return ly


def get_train_w(device, num_classes):
    w_train = torch.ones([num_classes], dtype=torch.float32, device=device)
    w_train.requires_grad = False
    return w_train

def get_val_w(device, num_val_samples):
    w_val=np.sum(num_val_samples)/num_val_samples
    w_val=w_val/np.linalg.norm(w_val)
    w_val=torch.tensor(w_val,dtype=torch.float32, device=device)
    w_val.requires_grad=False
    return w_val


# added by Bojian
def loss_adjust_cross_entropy_cdt(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*new_dy+new_ly
    else:
        x = logits*dy+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss


# added by Bojian
def loss_adjust_cross_entropy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*F.sigmoid(new_dy)+new_ly
    else:
        x = logits*F.sigmoid(dy)+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss


# added by Bojian
def cross_entropy(logits, targets, params=[], group_size=1):
    if len(params) == 3:
        return F.cross_entropy(logits, targets, weight=params[2])
    else:
        return F.cross_entropy(logits, targets)