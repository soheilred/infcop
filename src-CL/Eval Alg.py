"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import networks as net
import utils as utils
from prune import SparsePruner
from manager import Manager
import time
from torch.optim.lr_scheduler  import MultiStepLR

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch', choices=['vgg16', 'vgg16bn', 'resnet18', 'densenet121'], help='Architectures')
FLAGS.add_argument('--mode', choices=['t','c','p','e','all'], default='all', help='Run mode: train, calc. conns, prune, finetune, or evaluate')
FLAGS.add_argument('--dataset', type=str, choices=['pmnist', '6splitcifar', '11splitcifar'], default='6splitcifar', help='Name of dataset')
FLAGS.add_argument('--single_task', action='store_true', default=False, help='Run only the current task')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--num_outputs', type=int, default=-1, help='Num outputs for dataset')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--cores', type=int, default=4, help='Number of CPU cores.')
# Other.
# Paths.
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='', help='Location to save model')
# Training options.
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--batch_size', type=int, default=512, help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
FLAGS.add_argument('--Milestones', nargs='+', type=float, default=[30,60,90])
FLAGS.add_argument('--Gamma', type=float, default=0.1)   
FLAGS.add_argument('--train_bn', action='store_true', default=False, help='Train batch norm layers after task 0')
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='sparse', choices=['sparse'], help='Pruning method to use')
FLAGS.add_argument('--prune_filters', action='store_true', default=False, help='Average over filters when pruning and freezing')
FLAGS.add_argument('--reinit', action='store_true', default=False, help='Reininitialize pruned weights to non-zero values')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')
FLAGS.add_argument('--freeze_perc', type=float, default=0.0)                   
FLAGS.add_argument('--num_freeze_layers', type=int, default=0)     
FLAGS.add_argument('--freeze_order', choices=['top','bottom', 'random'], default=['top'],help='Order of selection for layer freezing, by connectivity')





###################################################################################################################################################
###
###     Main function
###
###################################################################################################################################################


def main():
    args = FLAGS.parse_args()
    ### Early termination conditions
    if args.prune_perc_per_layer <= 0:
        print("non-positive prune perc",flush = True)
        return    
    torch.cuda.set_device(0)
    
    if args.dataset == "6splitcifar":
        taskset = [*range(0,6,1)]
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0
     
    final_task_num = taskset[-1]
    print("Final task number: ", final_task_num)
    ###################
    ##### Prepare Ckpt
    ###################
    args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(final_task_num))

    finalpath = os.path.join(args.save_prefix, "final.pt")


    if os.path.isfile(finalpath) == False:
        print("Pruning wasn't finished, no trained checkpoint found for final task number: ", final_task_num)
        return 0
    else:
        ckpt = torch.load(finalpath)

    # pruner = SparsePruner(args, model, all_task_masks, composite_mask, conns, conn_aves, batchnorms)
    pruner = SparsePruner(args, ckpt)
    if args.cuda:
        pruner.model = pruner.model.cuda()

    ### Insert logic for looping over remaining tasks
    for task in taskset[args.task_num:]:
        print("Task Number: ", task)
        pruner.task_num = task

        ckpt = torch.load(finalpath)
        model = ckpt['model']
        pruner.model = model

        # Create the manager object.
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="train")
        test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="test")
        val_data_loader = test_data_loader
        pruner.testloader = test_data_loader

        ### This is for producing and setting the classifier layer for a given task's # classes
        pruner.model.set_dataset(str(task))
                
        manager = Manager(args, pruner, train_data_loader, val_data_loader, test_data_loader, task)
        print("Manager created")

        

        ### Evaluate performance on task
        if  args.mode == "e":
            print("Evaluating Accuracy")
            manager.eval(restore_bns = True)

        if args.single_task == True: 
            return 0

    

    
    
        

    
if __name__ == '__main__':
    main()
