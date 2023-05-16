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
# import networks as net
import utils as utils
from manager import Manager
import time
from torch.optim.lr_scheduler  import MultiStepLR

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--arch', choices=['vgg16', 'vgg16bn', 'resnet18', 'densenet121'], help='Architectures')
FLAGS.add_argument('--mode', choices=['t','c','p','e','all'], default='all', help='Run mode: train, calc. conns, prune, finetune, or evaluate')
FLAGS.add_argument('--dataset', type=str, choices=['cifar100subsets', '6splitcifar', '11splitcifar'], default='6splitcifar', help='Name of dataset')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--single_task', action='store_true', default=False, help='Run only the current task')
FLAGS.add_argument('--train_bn', action='store_true', default=False, help='Train batch norm layers after task 0')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
# Training options.
FLAGS.add_argument('--train_epochs', type=int, default=2, help='Number of epochs to train for')
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--batch_size', type=int, default=512, help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
FLAGS.add_argument('--Milestones', nargs='+', type=float, default=[30,60,90])
FLAGS.add_argument('--Gamma', type=float, default=0.1)   
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='structured', choices=['structured', 'unstructured'], help='Pruning method to use')
FLAGS.add_argument('--reinit', action='store_true', default=False, help='Reininitialize pruned weights to non-zero values')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')
FLAGS.add_argument('--finetune_epochs', type=int, default=2, help='Number of epochs to finetune for after pruning')





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
    
    ### Determines which tasks are included in the overall sequence
    if args.dataset == "6splitcifar":
        taskset = [*range(0,6,1)]
    elif args.dataset == "11splitcifar":
        taskset = [*range(0,11,1)]
    elif args.dataset == "cifar100subsets":
        taskset = [*range(0,6,1)]
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0
    
    
    
    ###################
    ##### Prepare Checkpoint and Manager
    ###################
    args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(args.task_num))
    os.makedirs(args.save_prefix, exist_ok = True)

    ### Path to load previous task's checkpoint, if not starting at task 0
    previous_task_path = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(args.task_num-1), "final.pt")
    ### Path to any intermediate trained checkpoint data if resuming mid-task
    trained_path = os.path.join(args.save_prefix, "trained.pt")

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    ckpt = None

    ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
    if os.path.isfile(previous_task_path) == True and (args.mode == "t" or args.mode == "all"):
        ckpt = torch.load(previous_task_path)
    elif os.path.isfile(trained_path) == True and (args.mode == "p" or args.mode == "c"):
        ckpt = torch.load(trained_path)
    else:
        print("No checkpoint file found")
        if args.task_num > 0:
            return 0

    ### Initialize the manager using the checkpoint.
    ### Manager handles pruning, connectivity calculations, and training/eval
    manager = Manager(args, ckpt)




    ###################
    ##### Loop Through Tasks
    ###################
    
    ### Logic for looping over remaining tasks
    for task in taskset[args.task_num:]:
        ### Update paths as needed for each new task
        args.save_prefix = os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str(task))
        os.makedirs(args.save_prefix, exist_ok = True)
        trained_path = os.path.join(args.save_prefix, "trained.pt")
        finetuned_path = os.path.join(args.save_prefix, "final.pt")

        ### Prepare dataloaders for new task
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="train")
        val_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="test")
        test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=task, set="test")
        manager.train_loader = train_data_loader
        manager.val_loader = val_data_loader
        manager.test_loader = test_data_loader

        ### This is for producing and setting the classifier layer for a given task's # classes
        manager.network.add_dataset(str(task), 10)
        manager.network.set_dataset(str(task))


        ### train for new task
        if  args.mode == "t" or args.mode == "all":
            print("Training", flush = True)
            manager.train(args.train_epochs, save=True, savename=trained_path)
                
        ### calculate connectivity scores
        if  args.mode == "c" or args.mode == "all":
            print("Calculate Connectivities")
            manager.calc_conns()
            utils.save_ckpt(manager=manager, savename=trained_path) 
            

        ### Prune unecessary weights or nodes
        if  args.mode == "p" or args.mode == "all":
            manager.prune()
            
            print('\nPost-prune eval:')
            errors = manager.eval()
            accuracy = 100 - errors[0]  # Top-1 accuracy.
            utils.save_ckpt(manager, finetuned_path)
    
            # Do final finetuning to improve results on pruned network.
            if args.finetune_epochs:
                print('Doing some extra finetuning...')
                manager.train(args.finetune_epochs, save=True, savename=finetuned_path, best_accuracy=0)
            
            print('-' * 16)
            print('Pruning summary:')
            manager.check(True)
            print('-' * 16)
            print("\n\n\n\n")

        ### Save the checkpoint and move on to the next task if required
        utils.save_ckpt(manager, savename=finetuned_path)
                    
        if args.single_task == False and args.mode == "all":
            manager.increment_task()
        else: 
            return 0


if __name__ == '__main__':
    main()
