"""Contains utility functions for calculating activations and connectivity. Adapted code is acknowledged in comments"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import DataGenerator as DG
from PIL import Image
import cifarsubsets
import network as net

import time
import copy
import math
import sklearn
import random 

import scipy.spatial     as ss

from math                 import log, sqrt
from scipy                import stats
from sklearn              import manifold
from scipy.special        import *
from sklearn.neighbors    import NearestNeighbors


visualisation = {}

"""
hook_fn(), activations(), and get_all_layers() adapted from: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
"""

#### Hook Function
def hook_fn(m, i, o):
    visualisation[m] = o 


### Create forward hooks to all layers which will collect activation state
def get_all_layers(net, hook_handles, item_key):
    if item_key != -1:
        for module_idx, module in enumerate(net.shared.modules()):
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                if(module_idx == item_key):
                    hook_handles.append(module.register_forward_hook(hook_fn))
    else:
        print("classifier hooks")
        hook_handles.append(net.classifier.register_forward_hook(hook_fn))


### Process and record all of the activations for the given pair of layers
def activations(data_loader, model, cuda, item_key):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []

    get_all_layers(model, handles, item_key)
    print('Collecting Activations for Layer %s'%(item_key))

    ### Note: The stacking will cause an error if the dataloader only has one batch, seemingly. Not currently an issue, but leaving this just in case
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            model(x_input.cuda())

            if temp_op is None:
                temp_op        = visualisation[list(visualisation.keys())[0]].cpu().numpy()
                temp_labels_op = y_label.numpy()

            else:
                temp_op        = np.vstack((visualisation[list(visualisation.keys())[0]].cpu().numpy(), temp_op))
                temp_labels_op = np.hstack((y_label.numpy(), temp_labels_op))

            if step % 100 == 0:
                if parents_op is None:
                    parents_op = copy.deepcopy(temp_op)
                    labels_op  = copy.deepcopy(temp_labels_op)

                    temp_op        = None
                    temp_labels_op = None

                else:
                    parents_op = np.vstack((temp_op, parents_op))
                    labels_op  = np.hstack((temp_labels_op, labels_op))

                    temp_op        = None
                    temp_labels_op = None


    if parents_op is None:
        parents_op = copy.deepcopy(temp_op)
        labels_op  = copy.deepcopy(temp_labels_op)

        temp_op        = None
        temp_labels_op = None

    else:
        parents_op = np.vstack((temp_op, parents_op))
        labels_op  = np.hstack((temp_labels_op, labels_op))

        temp_op        = None
        temp_labels_op = None

    # Remove all hook handles
    for handle in handles:
        handle.remove()    
    
    del visualisation[list(visualisation.keys())[0]]

    ### Average activations for a given filter
    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    return parents_op, labels_op





### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataloader(dataset, batch_size, num_workers=4, pin_memory=False, normalize=None, task_num=0, set="train"):
    
    
    # standard split CIFAR-10/100 sequence of tasks
    if dataset == "6splitcifar":
        dataset = cifarsubsets.get_splitCIFAR(task_num=task_num)
    # 6 custom subsets of CIFAR-100, each with 10 classes, designed to test task similarity
    elif dataset == "cifar100subsets":
        dataset = cifarsubsets.get_CIFARsubset(task_num=task_num)
    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        
    ### Makes a custom dataset for CIFAR through torch
    generator = DG.CifarDataGenerator(dataset[set]['x'],dataset[set]['y'])

    ### Loads the custom data into the dataloader
    if set == "train":
        return data.DataLoader(generator, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory=pin_memory)
    else:
        return data.DataLoader(generator, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)


### Saves a checkpoint of the model
def save_ckpt(manager, savename):
    """Saves model to file."""
    manager.get_batchnorm()

    # Prepare the ckpt.
    ckpt = {
        'args': manager.args,
        'batchnorms': manager.batchnorms,
        'all_task_masks': manager.all_task_masks,
        'conns' : manager.conns,
        'network': manager.network,
    }

    # Save to file.
    torch.save(ckpt, savename)






#####################################################
###    Masking Functions
#####################################################

### Get a binary mask where all previously frozen weights are indicated by a value of 1
### After pruning on the current task, this will still return the same masks, as the new weights aren't frozen until the task ends
def get_frozen_mask(weights, module_idx, all_task_masks, task_num):
    mask = torch.zeros(weights.shape)

    ### Include all weights used in past tasks (which would have been subsequently frozen)
    for i in range(0, task_num):
        if i == 0:
            mask = all_task_masks[i][module_idx].clone().detach()
        else:
            mask = torch.maximum(all_task_masks[i][module_idx], mask)
            
    return mask
        
    
### Get a binary mask where all unpruned, unfrozen weights are indicated by a value of 1
### Unlike get_frozen_mask(), this mask will change after pruning since the pruned weights are no longer trainable for the current task
def get_trainable_mask(module_idx, all_task_masks, task_num):

    mask = all_task_masks[task_num][module_idx].clone().detach()

    frozen_mask = get_frozen_mask(mask, module_idx, all_task_masks, task_num)
    
    mask[frozen_mask.eq(1)] = 0
    
    return mask
    
    
    