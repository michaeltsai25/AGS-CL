import os,sys
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import inf
import pandas as pd
from PIL import Image
from sklearn.feature_extraction import image
from arguments import get_args
from typing import Iterable, Optional
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
args = get_args()

#review this function
def gs_cal(t, x, y, criterion, model, sbatch=20):
    
    # Init
    param_R = {}
    
    for name, param in model.named_parameters():
        if len(param.size()) <= 1:
            continue
        name = name.split('.')[:-1]
        name = '.'.join(name)
        param = param.view(param.size(0), -1)
        param_R['{}'.format(name)]=torch.zeros((param.size(0)))
    
    # Compute
    model.train()

    for i in range(0,x.size(0),sbatch):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cpu()
        images=x[b]
        target=y[b]

        # Forward and backward
        outputs = model.forward(images, True)[t]
        cnt = 0
        
        for idx, j in enumerate(model.act):
            j = torch.mean(j, dim=0)
            if len(j.size())>1:
                j = torch.mean(j.view(j.size(0), -1), dim = 1).abs()
            model.act[idx] = j
            
        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.act[cnt].abs().detach()*sbatch
            cnt+=1 

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key]=(param_R[key]/x.size(0))
    return param_R

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################
def copy_model(model):
    for module_ in model.net:
        if isinstance(module_, torch.nn.ModuleList()):
            for linear_ in module_:
                linear_.clean()
        if isinstance(module_, torch.nn.ReLU()) or isinstance(module_, torch.nn.Linear()) or isinstance(module_, torch.nn.Conv2d()) or isinstance(module_, torch.nn.MaxPool2d()) or isinstance(module_, torch.nn.Dropout()):
            module_.clean()

    return deepcopy(model)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cpu()
        images=x[b]
        target=y[b]

        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images)[t]
        loss= criterion(outputs, target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/x.size(0)
    return fisher



########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

def clip_relevance_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)

    return total_norm

########################################################################################################################

import torch
from tqdm.auto import tqdm
import torch.nn as nn

from collections import defaultdict

from agents.exp_replay import Memory
from types import MethodType

import pytorch_lightning as pl
import wandb
import random
import models

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            return res[0]
        else:
            return res

def orthonormalize(vectors, gpu, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in tqdm(range(start_idx, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors

#TODO: run debugger to check code
def project_vec(model, omega, proj_basis, gpu):
    tot_grad = torch.empty(0)
    for name, param in model.named_parameters():
        if 'bias' in name or 'last' in name:
            continue
        key = name.split('.')[0]
        temp = torch.ones_like(param)
        for i, node in enumerate(param):
            grad_vec = parameters_to_grad_vector(node)
            if proj_basis.shape[1] > 0:  # param x basis_size
                dots = torch.matmul(grad_vec, proj_basis)  # basis_size
                # out = torch.matmul(proj_basis, dots)
                # TODO : Check !!!!
                out = torch.matmul(proj_basis, dots.T)
                temp[i] = grad_vec - (omega[key]*out)
            else:
                temp[i] = grad_vec - (omega[key]*torch.zeros_like(grad_vec))
        tot_grad = torch.cat(tot_grad, temp)
        
    return tot_grad

def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param


def validate(testloader, model, gpu, size):
    model.eval()

    acc = 0
    acc_cnt = 0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            if size is None or idx < size:
                data, target, task = data
                if gpu:
                    with torch.no_grad():
                        data = data.cuda()
                        target = target.cuda()

                outputs = model.forward(data, task)

                acc += accuracy(outputs, target)
                acc_cnt += 1

            else:
                break
    return acc / acc_cnt


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def get_n_trainable(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable