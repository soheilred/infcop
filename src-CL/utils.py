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
import splitCIFAR
import networks as net

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



# ### Calculate pearson correlation for a parent node and each downstream child node in the next layer
# ### p1 is a 1d array of activations while c1_op is a 2d array, with dimensions for the number of nodes and the number of activations
# def corr(p1, c1_op):
#     corrs_p = []
#     if (np.std(p1) < 0.001):
#         print("insufficient parent standard deviation of acts")
#         corrs_p = [0]*len(c1_op[0])
#     else:
#         for c in range(len(c1_op[0])):
#             ### This is to avoid errors with dividing by 0 with Pearson correlation
#             if (np.std(c1_op[:,c]) < 0.001):
#                 corrs_p.append(0) 
#             else:
#                 corrcoef_mat = np.corrcoef(p1, c1_op[:,c])
                
#                 corrs_p.append(abs(corrcoef_mat[0][1]))  
#     return np.asarray(corrs_p[:]) 
        



def get_dataloader(dataset, batch_size, num_workers=4, pin_memory=False, normalize=None, task_num=0, set="train"):
    if dataset == "6splitcifar":
        dataset = splitCIFAR.get(task_num=task_num)
    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        
    ### Makes a custom dataset for CIFAR through torch
    generator = DG.CifarDataGenerator(dataset[set]['x'],dataset[set]['y'])

    ### Loads the custom data into the dataloader
    return data.DataLoader(generator, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory=pin_memory)



### Produces an initialized model and initial shared mask prior to beginning training
def init_dump(arch,savepath):
    """Dumps pretrained model in required format."""
    if arch == 'vgg16':
        model = net.ModifiedVGG16()
    elif arch == 'resnet18':
        model = net.resnet18()
    else:
        raise ValueError('Architecture type not supported.')
    

    composite_mask = {}
    task_mask = {}
    all_task_masks = {}
    
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print("appending conv or linear layer")
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            task = torch.ByteTensor(module.weight.data.size()).fill_(0)
            
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
                task = task.cuda()
                
            task_mask[module_idx] = mask
            composite_mask[module_idx] = task
    

    all_task_masks[0] = [task_mask]
    
    
    torch.save({
        'all_task_masks': all_task_masks,
        'composite_mask': composite_mask,     
        'model': model,
        'conns' : {},
        'conn_aves' : {},
    }, savepath)


