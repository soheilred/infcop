"""
Handles all the pruning and connectivity. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import collections
import time
import copy
import torch
import random
import multiprocessing
import torch.nn as nn
import numpy             as np
import networks as net
import utils as utils

# Custom imports
# from utils import activations, corr
from utils import activations


class SparsePruner(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the pruner to explicitly show which arguments are used by it
    def __init__(self, args, checkpoint):


        self.args = args
        self.task_num = args.task_num
        self.testloader = None 
        

        if checkpoint != None:
            self.model = checkpoint['model']
            self.all_task_masks = checkpoint['all_task_masks']
            self.conns = checkpoint['conns']
            self.conn_aves = checkpoint['conn_aves']
            self.batchnorms = checkpoint['batchnorms']        
        else:
            if self.args.arch == 'resnet18':
                self.model = net.resnet18()
            else:
                raise ValueError('Architecture type not supported.')
            
            ### This is for producing and setting the classifier layer for a given task's # classes
            self.model.add_dataset(str(0), self.args.num_outputs)
            self.model.set_dataset(str(0))
            
            self.all_task_masks = {}
            self.batchnorms = {}
            self.conns = {}
            self.conn_aves = {}
            
            task_mask = {}
            batchnorm = {}
            
            for module_idx, module in enumerate(self.model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    # print("appending conv or linear layer")
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(1)

                    mask = mask.cuda()

                    task_mask[module_idx] = mask

                if isinstance(module, nn.BatchNorm2d):
                    batchnorm[module_idx] = {}
                    batchnorm[module_idx]["weight"] = module.weight.data.clone()
                    batchnorm[module_idx]["bias"] = module.bias.data.clone()
                    batchnorm[module_idx]["running_mean"] = module.running_mean.data.clone()
                    batchnorm[module_idx]["running_var"] = module.running_var.data.clone()
                    batchnorm[module_idx]["num_batches_tracked"] = module.num_batches_tracked.data.clone()
            
            self.batchnorms[0] = batchnorm
            self.all_task_masks[0] = [task_mask]
            
        
        
        print("#######################################################################")
        print("Finished Initializing Pruner")
        print("All task Masks keys: ", self.all_task_masks.keys())
        print("Batchnorms tasks: ", self.batchnorms.keys())
        print("Num outputs: ", self.args.num_outputs)
        print("Conns keys: ", self.conns.keys())
        print("Conn_aves keys: ", self.conn_aves.keys())
        print("Dataset: " + str(self.args.dataset))
        print("#######################################################################")

    
     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####
    #####  Use: Gets the connectivity between each pair of convolutional or linear layers. 
    #####       The primary original code for our published connectivity-based freezing method
    #####
    ###########################################################################################
    """

      
    def calc_conns(self):
        self.task_conns = {}
        self.task_conn_aves = {}        
        
        #!# Probably just calculate the activations here before the looping
        
        
        #!# This can still be the same, splitting by parents and children, it just needs to index the acts appropriately rather than generate them each time
        ### Record the indices of all adjacent pairs of layers in the shared network
        ### This numbering method reflects the "layer index" in Figs. 2-4 of the accompanying paper
        parents = []
        children = []
        i = 0
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if (i == 0):
                    parents.append(module_idx)
                    i += 1
                else:
                    parents.append(module_idx)
                    children.append(module_idx)
        children.append(-1)
        for key_id in range(0,len(parents)): 
            self.task_conn_aves[parents[key_id]], self.task_conns[parents[key_id]] = self.calc_conn([parents[key_id]], [children[key_id]], key_id)
            
        self.conn_aves[self.task_num] = self.task_conn_aves
        self.conns[self.task_num] = self.task_conns


   
    def calc_conn(self, parent_key, children_key, key_id):
        self.model.eval()
    
        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")
    
        p1_op = {}
        c1_op = {}
        p1_lab = {}
        c1_lab = {}    

        unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
        act         = {}
        lab         = {}
    

        ### Get activations and labels from the function in utils prior to calculating connectivities
        for item_key in unique_keys:
            act[item_key], lab[item_key] = activations(self.testloader, self.model, self.args.cuda, item_key)

        for item_idx in range(len(parent_key)):
            p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
            p1_lab[str(item_idx)] = copy.deepcopy(lab[parent_key[item_idx]]) 
            c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])
            c1_lab[str(item_idx)] = copy.deepcopy(lab[children_key[item_idx]])
        
        del act, lab

       
        print("----------------------------------")
        # print("Begin Execution of conn estimation")
    
        parent_aves = []
        p1_op = np.asarray(list(p1_op.values())[0])
        p1_lab = np.asarray(list(p1_lab.values())[0])
        c1_op = np.asarray(list(c1_op.values())[0])
        c1_lab = np.asarray(list(c1_lab.values())[0])
    
        task = self.task_num


        for label in list(np.unique(np.asarray(p1_lab))):
            # print("Parent mean and stdev: ", np.mean(p1_op[p1_lab == label]), " ", np.std(p1_op[p1_lab == label]))

            p1_op[p1_lab == label] -= np.mean(p1_op[p1_lab == label])
            p1_op[p1_lab == label] /= np.std(p1_op[p1_lab == label])



            c1_op[c1_lab == label] -= np.mean(c1_op[c1_lab == label])
            c1_op[c1_lab == label] /= np.std(c1_op[c1_lab == label])

        """
        Code for averaging conns by parent prior by layer
        """
        
        parent_class_aves = []
        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []
        
        # print("p1_op shape: ", p1_op.shape)

        for c in list(np.unique(np.asarray(p1_lab))):
            p1_class = p1_op[np.where(p1_lab == c)]
            c1_class = c1_op[np.where(c1_lab == c)]

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            coefs = np.corrcoef(p1_class, c1_class, rowvar=False)
            
            parents = []
            for i in range(0, len(p1_class[0])):
                parents.append(coefs[i, len(p1_class[0]):])
            parents = np.abs(np.asarray(parents))

            ### This is a growing list of each p-c connectivity for all activations of a given class
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)
        
        ### Averages all classes, since all class priors are the same for cifar10 and 100
        conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
        
        ### Then average over the parents and children to get the layer-layer connectivity
        layer_ave = np.mean(conn_aves)

        return layer_ave, conn_aves



    
    """
    ##########################################################################################################################################
    Pruning Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self):
        print('Pruning for dataset idx: %d' % (self.task_num))
        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.args.prune_perc_per_layer))
    
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                frozen_mask = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, self.task_num)
                print("Num frozen preprune: ", (frozen_mask.view(-1).eq(1).sum()/ frozen_mask.view(-1).numel()))
                ### Get the pruned mask for the current layer
                pruned_mask = self.pruning_mask(module.weight.data, frozen_mask, module_idx)
                # print("Num pruned preprune: ", (pruned_mask.view(-1).eq(1).sum()/ pruned_mask.view(-1).numel()))
                print("\n")
                # Set pruned weights to 0.
                module.weight.data[pruned_mask.eq(1)] = 0.0
                self.all_task_masks[self.task_num][0][module_idx][pruned_mask.eq(1)] = 0
               
      
 
    def pruning_mask(self, weights, frozen_mask, layer_idx):
        """
            Ranks prunable filters by magnitude. Sets all below kth to 0.
            Returns pruned mask.
        """

        weight_magnitudes = weights.abs()
        
        ### Setting to less-than will allow for fewer new weights to be frozen if frozen weights are deemed sufficient
        ###    With simple weight-based pruning though this should probably not be implemented


        ### The frozen mask has 1's indicating frozen weight indices
        if len(weights.size()) > 2 and self.args.prune_filters == True:
            ### Boolean mask for filters in dim 0 with True indicating unfrozen filters
            filter_frozen_mask = frozen_mask.eq(0).any(dim=3).any(dim=2).any(dim=1)
            filter_magnitudes = torch.mean(weight_magnitudes, axis=(1,2,3))
            ### Get only the trainable filters
            current_task_filters = filter_magnitudes[filter_frozen_mask]
            prune_rank = round(self.args.prune_perc_per_layer * current_task_filters.numel())
        else:
            weight_frozen_mask = frozen_mask.eq(0)
            current_task_weights = weight_magnitudes[weight_frozen_mask]
            prune_rank = round(self.args.prune_perc_per_layer * current_task_weights.numel())
        
        """
            Code for increasing the freezing percent of a given layer based on connectivity
        """

        prune_mask = torch.zeros(weights.shape)

        ### Set all weights under the magnitude threshold to be pruned
        if len(weights.size()) > 2 and self.args.prune_filters == True:
            prune_value = current_task_filters.view(-1).cpu().kthvalue(prune_rank)[0]
            filter_prune_mask = filter_magnitudes.abs().le(prune_value)
            expanded_prune_mask = filter_prune_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(prune_mask.shape)
            prune_mask[expanded_prune_mask]=1
        else:
            prune_value = current_task_weights.view(-1).cpu().kthvalue(prune_rank)[0]
            prune_mask[weight_magnitudes.abs().le(prune_value)]=1
            
        ### Prevent pruning of any previously frozen weights
        prune_mask[frozen_mask.eq(1)]=0
    
        ### Check how many weights are being chosen for pruning
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, prune_mask.eq(1).sum(), prune_mask.numel(),
              100 * prune_mask.eq(1).sum() / prune_mask.numel(), weights.numel()))

        return prune_mask
        
        
        
    
    """
    ##########################################################################################################################################
    Update Functions
    ##########################################################################################################################################
    """

    ### Reinitialize previously pruned weights
    # def reinitialize_pruned(self):
    #     print("Reinitializing pruned weights")
    #     for module_idx, module in enumerate(self.model.shared.modules()):
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             # if module_idx in [7,10]:
    #             ### Get the pruned mask for the current layer
    #             layer_comp_mask = self.composite_mask[module_idx]
    #             print("Shape before4: ", module.weight.data.size())
    #             module.weight.data[layer_comp_mask==self.task_num] = nn.init.normal_(module.weight.data[layer_comp_mask==self.task_num], mean=0.0, std=0.01)
    #             # module.weight.data[0:5,:,:,:] = nn.init.normal_(module.weight.data[0:5,:,:,:], mean=0.0, std=0.01)
    #             print("Shape after: ", module.weight.data.size())


    ### Set all frozen and pruned weights' gradients to zero for training
    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        # assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                trainable_mask = utils.get_trainable_mask(module_idx, self.all_task_masks, self.task_num)
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[trainable_mask.eq(0)] = 0
                if self.task_num>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)
                    
            elif 'BatchNorm' in str(type(module)) and self.task_num>0 and self.args.train_bn == False:
                    # Set grads of batchnorm params to 0.
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    ### Applies appropriate mask to recreate task model for inference
    def apply_mask(self):
        """To be done to retrieve weights just for a particular dataset"""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data[self.all_task_masks[self.task_num][0][module_idx].eq(0)] = 0.0






    ### Reload a previously saved state of the network's batchnorm layers
    def restore_batchnorm(self,taskid):
        """Use the given biases to replace existing biases."""
        print("Restoring BN for taskid: ", taskid)
        with torch.no_grad():
            for module_idx, module in enumerate(self.model.shared.modules()):
                if isinstance(module, nn.BatchNorm2d):
                                module.weight.copy_(((self.batchnorms[taskid])[module_idx])["weight"])
                                module.bias.copy_(self.batchnorms[taskid][module_idx]["bias"])
                                module.running_mean.copy_(self.batchnorms[taskid][module_idx]["running_mean"])
                                module.running_var.copy_(self.batchnorms[taskid][module_idx]["running_var"])
                                module.num_batches_tracked.copy_(self.batchnorms[taskid][module_idx]["num_batches_tracked"])                

    ### Save the current batchnorm layers' states.
    def get_batchnorm(self):
        """Gets a copy of the current batchnorms."""
        batchnorm = {}
        print("Saving batchnorms for task: ", self.task_num)
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.BatchNorm2d):
                batchnorm[module_idx] = {}
                batchnorm[module_idx]["weight"] = module.weight.data.clone()
                batchnorm[module_idx]["bias"] = module.bias.data.clone()
                batchnorm[module_idx]["running_mean"] = module.running_mean.data.clone()
                batchnorm[module_idx]["running_var"] = module.running_var.data.clone()
                batchnorm[module_idx]["num_batches_tracked"] = module.num_batches_tracked.data.clone()
                    
        self.batchnorms[self.task_num] = batchnorm






    """
        Turns previously pruned weights into trainable weights for
        current dataset.
    """
    ### Replace with task_increment function
    def increment_task(self):
        self.task_num += 1
                
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                task = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    task = task.cuda()
                task_mask[module_idx] = task

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[self.task_num] = [task_mask]

        print("Exiting finetuning mask")
