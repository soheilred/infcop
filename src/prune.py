import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.init as init
import pickle
import sys

import utils
import plot_tool
from data_loader import Data
from network import Network, train, test
from correlation import Activations
import constants as C
import logging
import logging.config

log = logging.getLogger("sampleLogger")

class Controller:
    def __init__(self, args):
        """Control the IMP's connectivity.
        """
        self.c_type = args.control_type
        self.c_layers = args.control_at_layer
        self.c_iter = args.control_at_iter
        self.c_epoch = args.control_at_epoch


class Pruner:
    def __init__(self, args, model, train_dataloader=None, test_dataloader=None,
                 controller=None, correlation=None): 
        """Prune the network.
        Parameters
        ----------
        model: Network
            The network to be pruned
        prune_percent: int
            The percent to which each layer of the network is pruned
        train_dataloader: dataloader
        test_dataloader: dataloader
        composite_mask:
            The composite mask stores the task number for which every weight was
            frozen, or if they are unfrozen the number is the current task 
        all_task_masks: dict
            A dictionary of binary masks, 1 per task, indicating
            which weights to include when evaluating that task 
        """
        self.model = model
        self.mask = None
        self.args = args
        self.prune_perc = args.prune_perc_per_layer * 100
        self.corrs = []
        self.controller = controller
        self.correlation = correlation
        self.num_layers = 0
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.comp_level = np.zeros(args.imp_total_iter, float)
        self.all_acc = np.zeros([args.imp_total_iter, args.train_epochs], float)
        self.init_state_dict = None
        # self.init_dump()

    def weight_init(self, m):
        '''Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    def init_lth(self):
        """Prepare the lth object by:
        1. initializing the network's weights
        2. saving the initial state of the network into the object
        3. saving the initial state model on the disk
        4. initializing the masks according to the layers size
        """

        # Weight Initialization
        if self.args.pretrained == "False":
            self.model.apply(self.weight_init)

        # Copying and Saving Initial State
        self.init_state_dict = copy.deepcopy(self.model.state_dict())
        run_dir = utils.get_run_dir(self.args)
        utils.save_model(self.model, run_dir, "initial_model.pth.tar")

        # Making Initial Mask
        self.init_mask()
        return self.init_state_dict

    def count_layers(self):
        count = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                count = count + 1
        return count

    def init_mask(self):
        """Make an empty mask of the same size as the model."""
        self.num_layers = self.count_layers()
        self.mask = [None] * self.num_layers
        layer_id = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[layer_id] = np.ones_like(tensor)
                # self.mask[layer_id] = torch.ones_like(param.data)
                layer_id += 1


    def reset_weights_to_init(self, initial_state_dict):
        """Reset the remaining weights in the network to the initial values.
        """
        step = 0
        mask_temp = self.mask
        for name, param in self.model.named_parameters():
            if "weight" in name:
                weight_dev = param.device
                param.data = torch.from_numpy(mask_temp[step] *
                                              initial_state_dict[name].
                                              cpu().numpy()).to(weight_dev)
                step = step + 1
            if "bias" in name:
                param.data = initial_state_dict[name]

    def prune_by_correlation(self, corr_con):
        correlation, layer_idx = corr_con
        # Calculate percentile value
        layer_id = 0
        # for name, param in self.model.named_parameters():

        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if layer_id in layer_idx:
                    # We do not prune bias term
                    weight = module[1].weight.data
                    weight = weight.cpu().numpy()
                    weight_dev = module[1].weight.device

                    kernel_size = self.mask[layer_id].shape[-1]
                    w_corr = np.tile(correlation[layer_id], reps=(kernel_size, kernel_size, 1, 1)).\
                                        transpose(3, 2, 1, 0)

                    # tensor = param.data.cpu().numpy()
                    alive = tensor[np.nonzero(weight)] # flattened array of nonzero values
                    percentile_value = np.percentile(abs(correlation[layer_id]), self.prune_perc)

                    # Convert Tensors to numpy and calculate
                    new_mask = np.where(abs(correlation[layer_id]) < percentile_value, 0,
                                        self.mask[layer_id])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    self.mask[layer_id] = new_mask
            layer_id += 1

    def prune_by_percentile(self):
        # Calculate percentile value
        layer_id = 0
        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), self.prune_perc)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0,
                                    self.mask[layer_id])
                new_mask = np.zeros_like(tensor) #.shape, dtype='float32')

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[layer_id] = new_mask
                layer_id += 1

    def prune_once(self, initial_state_dict, corr_con=None):

        if corr_con != None:
            self.prune_by_correlation(corr_con)
        else:
            self.prune_by_percentile()
        self.reset_weights_to_init(initial_state_dict)

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0.
            During training this is called to avoid storing gradients for the
            frozen weights, to prevent updating.
            This is unaffected in the shared masks since shared weights always
            have the current index unless frozen 
        """
        # assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.task_num)] = 0
                if self.task_num>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)
                    
            elif 'BatchNorm' in str(type(module)) and self.task_num>0:
                    # Set grads of batchnorm params to 0.
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Set all pruned weights to 0.
            This is just a prune() but with pre-calculated masks
        """
        # assert self.current_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                module.weight.data[layer_mask.gt(self.task_num)] = 0.0

    def apply_mask(self):
        """Applies appropriate mask to recreate task model for inference.
            To be done to retrieve weights just for a particular dataset
        """
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = -100

                # Any weights which weren't frozen in one of the tasks before
                # or including task # dataset_idx are set to 0 
                for i in range(0, self.task_num + 1):
                    if i == 0:
                        mask = self.all_task_masks[i][0][module_idx].cuda()
                    else:
                        mask = mask.logical_or(self.all_task_masks[i][0][module_idx].cuda())
                weight[mask.eq(0)] = 0.0

        self.model.eval()
   
    def control(self, corr, layers_dim, imp_iter):
        control_corrs = self.corrs + [corr]
        log.debug(f"apply controller at layer {self.controller.c_layers}")

        # get the weights from previous iteration
        prev_iter_weights = self.get_prev_iter_weights(imp_iter)

        # get connectivity
        connectivity = [(np.mean(control_corrs[imp_iter - 1][i]) /
                        (layers_dim[i][0] * layers_dim[i + 1][0]))
                        for i in range(len(layers_dim) - 1)]

        # get the coefficient based on connectivity
        for ind in self.controller.c_layers:
            prev_corr = self.get_prev_iter_correlation(control_corrs, layers_dim,
                                                         imp_iter, ind)
            prev_weight = prev_iter_weights[ind]

            # type 1
            if (self.controller.c_type == 1):
                control_weights = abs(prev_corr) / max(connectivity)

            # type 2
            elif (self.controller.c_type == 2):
                control_weights = torch.mul(prev_corr, prev_weight)

            # type 3
            elif (self.controller.c_type == 3):
                control_weights = 100 * abs(connectivity[ind]) / max(connectivity) # * prev_weight

            # type 4
            elif (self.controller.c_type == 4):
                control_weights = abs(prev_corr)
                control_weights = np.exp(control_weights) /\
                    np.exp(control_weights).sum()

            elif (self.controller.c_type == 5):
                control_weights = np.exp(abs(prev_corr))

            self.apply_controller(control_weights, ind)

    def apply_controller(self, control_weights, layer_idx):
        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (idx == layer_idx):
                    # weight = module[1].weight.detach().cpu().numpy()
                    weight = module[1].weight.data
                    print("network's weight shape", weight.shape)
                    mod_weight = weight.cpu().numpy()
                    weight_dev = module[1].weight.device
                    # control_weights = torch.from_numpy(control_weights.astype("float32")).to(weight_dev)
                    new_weight = torch.from_numpy((mod_weight * control_weights).astype("float32")).to(weight_dev)
                    # module[1].weight = torch.nn.Parameter(new_weight,
                    #                                        dtype=torch.float,
                    #                                        device=weight_dev)
                    print("control weight", np.linalg.norm(control_weights))
                    print("old weight", torch.linalg.norm(weight))
                    print("new weight", torch.linalg.norm(new_weight))
                    weight = new_weight
                    break
                idx += 1


    def get_prev_iter_correlation(self, control_corrs, layers_dim, imp_iter, ind):
        # the + 1 is for matching to the connectivity's dimension
        # weights = control_corrs[imp_iter - 1][ind - 1]
        weights = control_corrs[0][ind - 1]
        print("controller weight shape", weights.shape)
        kernel_size = layers_dim[ind][-1]
        # weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
        #                        transpose(1, 2).transpose(0, 3)
                               # transpose(1, 2).transpose(0, 3).transpose(0, 1)
        weights = np.tile(weights, reps=(kernel_size, kernel_size, 1, 1)).\
                               transpose(3, 2, 1, 0)
        print("controller weight shape", weights.shape)
        return weights


    def get_prev_iter_weights(self, imp_iter):
        run_dir = utils.get_run_dir(self.args)
        model = torch.load(run_dir + str(imp_iter) + '_model.pth.tar')
        # model.eval()
        weights = {}

        idx = 0
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (idx in self.controller.c_layers):
                    weights[idx] = module[1].weight
                idx += 1
        return weights


def perf_lth(logger, device, args, controller):
    ITERATION = args.imp_total_iter               # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.arch, num_classes, args.pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # warm up the pretrained model
    acc, _ = train(model, train_dl, loss_fn, optimizer, args.warmup_train, device)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    init_state_dict = pruning.init_lth()
    connectivity = []

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Nonzero percentage: {comp_level}")

        # Training the network
        for train_iter in range(args.train_epochs):

            # Training
            logger.debug(f"Training iteration {train_iter} / {args.train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, 
                              args.train_per_epoch, device)

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller after some epochs and some iterations
            if (train_iter == controller.c_epoch) and \
                (imp_iter in controller.c_iter):
                act = Activations(model, test_dl, device, args.batch_size)
                # corr = act.get_corrs()
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                             weight_decay=1e-4)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # if (imp_iter <= controller.c_iter):
        activations = Activations(model, test_dl, device, args.batch_size)
        # pruning.corrs.append(activations.get_corrs())
        pruning.corrs.append(activations.get_correlations())
        connectivity.append(activations.get_conns(pruning.corrs[imp_iter]))
        # utils.save_vars(corrs=pruning.corrs, all_accuracies=pruning.all_acc)

    return pruning.all_acc, connectivity
    
def perf_correlation_lth(logger, device, args, controller):
    ITERATION = args.imp_total_iter               # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.arch, num_classes, args.pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # warm up the pretrained model
    acc, _ = train(model, train_dl, loss_fn, optimizer, args.warmup_train, device)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    init_state_dict = pruning.init_lth()
    connectivity = []

    for imp_iter in tqdm(range(ITERATION)):
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            act = Activations(model, test_dl, device, args.batch_size)
            corr = act.get_correlations()
            layers_idx = act.get_layers_idx()
            pruning.prune_once(init_state_dict, corr_con=[corr, layers_idx])
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in range(args.train_epochs):

            # Training
            logger.debug(f"Training iteration {train_iter} / {args.train_epochs}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, 
                              args.train_per_epoch, device)

            # Test and save the most accurate model
            accuracy = test(model, test_dl, loss_fn, device)

            # apply the controller after some epochs and some iterations
            if (train_iter == controller.c_epoch) and \
                (imp_iter in controller.c_iter):
                act = Activations(model, test_dl, device, args.batch_size)
                # corr = act.get_corrs()
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                             weight_decay=1e-4)

            pruning.all_acc[imp_iter, train_iter] = accuracy

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        # if (imp_iter <= controller.c_iter):
        activations = Activations(model, test_dl, device, args.batch_size)
        # pruning.corrs.append(activations.get_corrs())
        pruning.corrs.append(activations.get_correlations())
        connectivity.append(activations.get_conns(pruning.corrs[imp_iter]))
        # utils.save_vars(corrs=pruning.corrs, all_accuracies=pruning.all_acc)

    return pruning.all_acc, connectivity
    
def eff_lth(logger, device, args, controller):
    ITERATION = args.imp_total_iter               # 35 was the default
    run_dir = utils.get_run_dir(args)
    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    num_classes = data.get_num_classes()

    network = Network(device, args.arch, num_classes, args.pretrained)
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    pruning = Pruner(args, model, train_dl, test_dl, controller)
    init_state_dict = pruning.init_lth()
    connectivity = []
    all_acc = []
    train_iter = np.zeros(ITERATION, int)
    max_acc = 1

    for imp_iter in tqdm(range(ITERATION)):
        accuracy = -1
        acc_list = []
        # except for the first iteration, cuz we don't prune in the first iteration
        if imp_iter != 0:
            pruning.prune_once(init_state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop")

        # Print the table of Nonzeros in each layer
        # comp_level = utils.print_nonzeros(model)
        pruning.comp_level[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training loop
        while (train_iter[imp_iter] < 30):
            if train_iter[imp_iter] > controller.c_epoch:
                if (accuracy > args.acc_thrd * max_acc / 100.0):
                    break

            # Training
            logger.debug(f"Accuracy {accuracy:.2f} at training iteration "
                         f"{train_iter[imp_iter]}, thsd: "
                         f"{args.acc_thrd * max_acc / 100.0}")
            acc, loss = train(model, train_dl, loss_fn, optimizer, 
                              args.train_per_epoch, device)

            # Test and save the most accurate model
            logger.debug("Testing...")
            accuracy = test(model, test_dl, loss_fn, device)
            acc_list.append(accuracy)

            # apply the controller after some epochs and some iterations
            if (train_iter[imp_iter] == controller.c_epoch) and \
                (imp_iter == controller.c_iter):
                act = Activations(model, test_dl, device, args.batch_size)
                corr = act.get_correlations()
                pruning.control(corr, act.layers_dim, imp_iter)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                             weight_decay=1e-4)

            # increment the training iterator
            train_iter[imp_iter] += 1

        all_acc.append(acc_list)
        logger.debug(all_acc)
        max_acc = max(all_acc[0])

        # Save model
        utils.save_model(model, run_dir, f"{imp_iter + 1}_model.pth.tar")

        # Calculate the connectivity
        activations = Activations(model, test_dl, device, args.batch_size)
        pruning.corrs.append(activations.get_correlations())
        connectivity.append(activations.get_conns(pruning.corrs[imp_iter]))
        # utils.save_vars(corrs=pruning.corrs, all_accuracies=pruning.all_acc)

    return all_acc, connectivity
    
def run_experiment(logger, args, device, run_dir):
    logger.debug(f"####### In {args.experiment_type} experiment #######")
    controller = Controller(args)
    acc_list = []
    conn_list = []

    for i in range(args.num_trial):
        logger.debug(f"In experiment {i} / {args.num_trial}")
        if args.experiment_type == "performance":
            all_acc, conn = perf_lth(logger, device, args, controller)

        elif args.experiment_type == "efficiency":
            all_acc, conn = eff_lth(logger, device, args, controller)

        elif args.experiment_type == "pcorr":
            all_acc, conn = perf_correlation_lth(logger, device, args, controller)

        acc_list.append(all_acc)
        conn_list.append(conn)
        utils.save_vars(save_dir=run_dir+str(i)+"_" , conn=conn,
                        all_accuracies=all_acc)
        # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + str(i) +
        #                             "all_accuracies")

    # all_acc = np.mean(acc_list, axis=0)
    # conn = np.mean(conn_list, axis=0)
    # plot_tool.plot_all_accuracy(all_acc, C.OUTPUT_DIR + "all_accuracies")
    # utils.save_vars(save_dir=run_dir, conn=conn, all_accuracies=all_acc)
    utils.save_vars(save_dir=run_dir, conn=conn_list, all_accuracies=acc_list)

def main():
    args = utils.get_args()
    logger = utils.setup_logger_dir(args)
    args = utils.get_yaml_args(args)
    device = utils.get_device(args)
    run_dir = utils.get_run_dir(args)
    run_experiment(logger, args, device, run_dir)


if __name__ == '__main__':
    main()
