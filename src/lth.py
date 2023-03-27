import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
# import seaborn as sns
import torch.nn.init as init
import pickle

import utils
from data_loader import Data
from network import Network, train, test
from correlation import Activations
import constants as C
import logging
import logging.config

log = logging.getLogger("sampleLogger")

class LTPruning:
    def __init__(self, model, arch_type, prune_percent, train_dataloader, test_dataloader):
        "docstring"
        self.model = model
        self.arch_type = arch_type
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.mask = None
        self.start_iter = 0
        self.lr = 1.2e-3
        self.end_iter = 100
        self.prune_type = "lt"
        self.prune_percent = prune_percent
        self.reinit = False
        self.num_layers = 0

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

    def count_layers(self):
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                step = step + 1
        return step

    def init_mask(self):
        """Make an empty mask of the same size as the model."""
        self.num_layers = self.count_layers()
        self.mask = [None] * self.num_layers
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[step] = np.ones_like(tensor)
                step = step + 1

    def prune_by_percentile(self):
        # Calculate percentile value
        step = 0
        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), self.prune_percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0,
                                    self.mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[step] = new_mask
                step += 1

    def original_initialization(self, initial_state_dict):
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
        # step = 0

    def prune_once(self, initial_state_dict):
        step = 0
        self.prune_by_percentile()
        # if the reinit option is activated
        if self.reinit:
            self.model.apply(self.weight_init)
            step = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy() *
                                                  self.mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
        else:
            # todo: why do we need an initialization here?
            self.original_initialization(initial_state_dict)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    def find_unstable_layers(self, control_corrs):
        control_corrs = np.array(control_corrs[1])
        # mid = abs(np.median(control_corrs[1]))
        thrsh = 100
        unstable_layers = np.where(np.abs(control_corrs) > thrsh)[0]
        return unstable_layers


    def controller(self, control_corrs, layers_dim, cont_type, imp_iter, cont_layer_list=None):
        # calculate the control proportional coefficient
        log.debug(f"apply controller at layer {cont_layer_list}")

        # coef = control_corrs[0][cont_layer_list] / max(control_corrs[1][layer_num], eps)
        if cont_layer_list is None:
            cont_layer_list = self.find_unstable_layers(control_corrs)

        #     log.debug(f"controller coef at layer {ind}: {coef}")
        #     contr_mask[ind] = (control_weights[ind] * coef).astype("float32")

        import ipdb; ipdb.set_trace()
        # get the weights from previous iteration
        prev_iter_weights = self.get_prev_iter_weights(imp_iter, cont_layer_list)

        # get connectivity
        connectivity = [(torch.mean(control_corrs[imp_iter - 1][i]).item() /
                        (layers_dim[i][0] * layers_dim[i + 1][0]))
                        for i in range(len(layers_dim) - 1)]

        # get the coefficient based on connectivity
        for ind in cont_layer_list:
            prev_corr = self.get_prev_iter_correlation(control_corrs, layers_dim,
                                                         imp_iter, ind)
            prev_weight = prev_iter_weights[ind]

            # type 1
            if (cont_type == 1):
                control_weights = prev_corr

            # type 2
            elif (cont_type == 2):
                control_weights = torch.mul(prev_corr, prev_weight)

            # type 3
            elif (cont_type == 3):
                control_weights = connectivity[ind] * prev_weight

            self.apply_controller(control_weights, ind)

        # self.apply_controller(control_weights=control_corrs, layer_list=layer_list)


    def get_prev_iter_correlation(self, control_corrs, layers_dim, imp_iter, ind):
        # the + 1 is for matching to the connectivity's dimension
        weights = control_corrs[imp_iter - 1][ind - 1]
        kernel_size = layers_dim[ind][-1]
        weights = weights.tile(dims=(kernel_size, kernel_size, 1, 1)).\
                               transpose(1, 2).transpose(0, 3)
                               # transpose(1, 2).transpose(0, 3).transpose(0, 1)
        return weights


    def get_prev_iter_weights(self, imp_iter, layers_list):
        model = torch.load(C.MODEL_ROOT_DIR + str(imp_iter) + '_model.pth.tar')
        # model.to(device)
        model.eval()
        weights = {}

        ind = 0
        for name, param in model.named_parameters():
            if ("weight" in name and 
               ("conv" in name or "fc" in name or "features" in name)):
                if ind in layers_list:
                    log.debug(f"weights at layer {ind} in iteration {imp_iter} is added")
                    weights[ind] = param.data
                ind += 1
            if ind > max(layers_list):
                break

        return weights


    def apply_controller(self, control_weights, layer_ind):
        ind = 0
        # get a handle to the layer's weights
        for name, param in self.model.named_parameters():
            # if "weight" in name:
            if ("weight" in name and 
               ("conv" in name or "fc" in name or "features" in name)):
                if ind == layer_ind:
                    # weight = param.data.cpu().numpy()
                    weight = param.data
                    # exp = np.exp(weight)
                    weight_dev = param.device
                    # contr_mask = (np.ones(weight.shape) * coef).astype("float32")
                    # param.data = torch.from_numpy(weight * exp * contr_mask).to(weight_dev)
                    # new_weights = utils.batch_mul(weight, control_weights)
                    new_weights = torch.mul(weight, control_weights)
                    param.data = new_weights.to(weight_dev)
                    # param.data = torch.from_numpy(weight * control_weights).to(weight_dev)
                    break
                ind += 1
        # weight = self.model.features[layer_list].weight.data.cpu().numpy()
        # cur_weights = layer_param.data.cpu().numpy()

def main():
    # preparing the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())

    # setting hyperparameters
    batch_size = 256
    num_epochs = 1

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = 3               # 35 was the default
    end_iter = 50               # 100 is the default
    start_iter = 0
    prune_percent = 10
    prune_type = "lt"
    lr = 1.2e-3
    print_freq = 1
    valid_freq = 1

    # arch_type = "vgg11"
    arch_type = "vgg16"
    # arch_type = "resnet"
    # arch_type = "alexnet"
    pretrained = True
    # pretrained = False
    dataset = "CIFAR10"
    MODEL_DIR = MODEL_ROOT_DIR + arch_type + "/" + dataset + "/"

    # Copying and Saving Initial State
    network = Network(device, arch_type, pretrained)
    preprocess = network.preprocess
    data = Data(batch_size, DATA_DIR, transform=preprocess)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    model = network.set_model()

    corr = []

    pruning = LTPruning(model, arch_type, prune_percent, train_dataloader, test_dataloader)
    # Weight Initialization
    model.apply(pruning.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.save_model(model, MODEL_DIR, "initial_state_dict.pth.tar")

    # Making Initial Mask
    pruning.make_mask(model)

    # Optimizer and Loss
    # TODO: why there are two criterion definitions?
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(end_iter, float)
    all_accuracy = np.zeros(end_iter, float)
    reinit = False

    # Iterative Magnitude Pruning main loop
    for _ite in range(start_iter, ITERATION):
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        if not _ite == 0:
            pruning.prune_by_percentile()
            # if the reinit option is activated
            if reinit:
                model.apply(pruning.weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy()
                                                      * pruning.mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                # todo: why do we need an initialization here?
                pruning.original_initialization(initial_state_dict)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: ---")
        logger.debug(f"[{_ite}/{ITERATION}] " + "IMP loop")

        ###################
        # model = network.set_model()

        # torch.save(model, MODEL_DIR + arch_type + str(_ite) + '-model.pt')
        # logger.debug('model is saved...!')
        ###################

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[_ite] = comp_level
        pbar = tqdm(range(end_iter))

        # Training the network
        for iter_ in pbar:
            logger.debug(f"{iter_}/{end_iter}" + " inside training loop " + arch_type)

            # Test and save the most accurate model
            if iter_ % valid_freq == 0:
                logger.debug("Testing...")
                accuracy = test(model, test_dataloader, criterion, device)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.save_model(model, MODEL_DIR, f"{_ite}_model_{prune_type}.pth.tar")

            # Training
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              epochs=num_epochs, device=device)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{end_iter} \n'
                    f'Loss: {loss:.6f} Accuracy: {accuracy:.2f}% \n'
                    f'Best Accuracy: {best_accuracy:.2f}%\n')

        # Calculate the connectivity
        activations = Activations(model, test_dataloader, device, batch_size)
        corr.append(activations.get_correlation())
        # Save the activations
        pickle.dump(corr, open(OUTPUT_DIR + arch_type + "_correlation.pkl", "wb"))

        # save the best model
        # writer.add_scalar('Accuracy/test', best_accuracy, comp_level)
        bestacc[_ite] = best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed
        # only for every {args.valid_freq} iterations. Therefore Accuracy saved
        # is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        # plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss")
        # plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy")
        # plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})")
        # plt.xlabel("Iterations")
        # plt.ylabel("Loss and Accuracy")
        # plt.legend()
        # plt.grid(color="gray")
        # plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp_level}.png", dpi=1200) 
        # plt.close()

        # Dump Plot values
        all_loss.dump(OUTPUT_DIR + f"{prune_type}_all_loss_{comp_level}.dat")
        all_accuracy.dump(OUTPUT_DIR + f"{prune_type}_all_accuracy_{comp_level}.dat")


        # Dumping mask
        with open(OUTPUT_DIR + f"{prune_type}_mask_{comp_level}.pkl", 'wb') as fp:
            pickle.dump(pruning.mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(end_iter, float)
        all_accuracy = np.zeros(end_iter, float)

    # Dumping Values for Plotting
    comp.dump(OUTPUT_DIR + f"{prune_type}_compression.dat")
    bestacc.dump(OUTPUT_DIR + f"{prune_type}_bestaccuracy.dat")

    # Plotting
    # a = np.arange(prune_iterations)
    # plt.plot(a, bestacc, c="blue", label="Winning tickets")
    # plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})")
    # plt.xlabel("Unpruned Weights Percentage")
    # plt.ylabel("test accuracy")
    # plt.xticks(a, comp, rotation ="vertical")
    # plt.ylim(0,100)
    # plt.legend()
    # plt.grid(color="gray")
    # utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    # plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    # plt.close()


if __name__ == '__main__':
    main()
