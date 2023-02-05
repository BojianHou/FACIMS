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
import torch.nn.functional as F  # added by Bojian
from torch.autograd import grad  # added by Bojian
import copy

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


# freeze and activate gradient w.r.t. parameters
def model_freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def model_activate(model):
    for param in model.parameters():
        param.requires_grad = True

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


def loss_adjust_cross_entropy_cdt(logits, targets, params):
    dy = params[0]
    ly = params[1]

    new_logits = logits*dy+ly

    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(new_logits, targets, weight=wy)
    else:
        loss = F.cross_entropy(new_logits, targets)
    return loss


def loss_adjust_cross_entropy(logits, targets, params):
    dy = params[0]
    ly = params[1]
    # logits = torch.Tensor.double(logits)
    # targets = torch.LongTensor(targets)
    # new_logits = logits * torch.sigmoid(dy) + ly
    new_logits = logits + ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(new_logits, targets, weight=wy)
    else:
        loss = F.cross_entropy(new_logits, targets)
    return loss


def loss_adjust_cross_entropy_manual(logits, targets, param):
    # print('new_logits = logits + torch.log(params)')
    new_logits = logits - torch.log(param)
    # new_logits = logits - params
    # new_logits = logits / params
    loss = F.cross_entropy(new_logits, targets)
    return loss


def cross_entropy(logits, targets, params=[], group_size=1):
    if len(params) == 3:
        return F.cross_entropy(logits, targets, weight=params[2])
    else:
        return F.cross_entropy(logits, targets)


def gather_flat_grad(loss_grad):
    #cnt = 0
    # for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    # g_vector
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])


def neumann_hyperstep_preconditioner(mean_d_L_up_d_post,  # mean over all the d_L_up_d_post,
                                     # as the vector for the use of Jacobian-Vector product
                                     list_d_L_low_d_post,
                                     elementary_lr,
                                     num_neumann_terms,
                                     list_post_model, prm):
    # list_d_L_low_d_post = [d_L_low_d_post.detach() for d_L_low_d_post in list_d_L_low_d_post]
    # list_d_L_low_d_post = copy.deepcopy(list_d_L_low_d_post)
    # list_post_model = copy.deepcopy(list_post_model)
    for post_model in list_post_model:
        model_activate(post_model)
    preconditioner = mean_d_L_up_d_post.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-(inverse Hessian) product
    i = 0

    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        old_counter = counter
        # hessian_term here is actually the sum of the product of
        # (1) the second derivative of post model w.r.t. L_low
        # (2) and the derivative of post model w.r.t. L_up
        hessian_term = torch.zeros(len(list_d_L_low_d_post[0]), device=prm.device)
        for d_L_low_d_post, post_model in zip(list_d_L_low_d_post, list_post_model):
            # torch.autograd.set_detect_anomaly(True)
            hessian_term += gather_flat_grad(
                           grad(d_L_low_d_post, post_model.parameters(),
                           grad_outputs=counter.view(-1), retain_graph=True, create_graph=True, allow_unused=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def get_trainable_hyper_params(params):
    return[param for param in params if param.requires_grad]


def assign_gradient(params, gradient, num_classes):
    i = 0
    for para in params:
        if para.requires_grad:
            num = para.nelement()
            grad = gradient[i:i+num].clone()
            grad = torch.reshape(grad, para.shape)
            para.grad = grad.clone()
            i += num
            # para.grad=gradient[i:i+num].clone()
            # para.grad=gradient[i:i+num_classes].clone()
            # i+=num_classes