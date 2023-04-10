import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets
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

class Pruner:
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
        self.init_state_dict = None
        # Weight Initialization
        self.model.apply(self.weight_init)
        # Making Initial Mask
        self.init_mask()

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

    def initialize_lth(self):
        """Prepare the lth object by:
        1. initializing the network's weights
        2. saving the initial state of the network into the object
        3. saving the initial state model on the disk
        4. initializing the masks according to the layers size
        """

        # Weight Initialization
        self.model.apply(self.weight_init)

        # Copying and Saving Initial State
        self.init_state_dict = copy.deepcopy(self.model.state_dict())
        utils.save_model(self.model, C.MODEL_ROOT_DIR, "/initial_model.pth.tar")

        # Making Initial Mask
        self.init_mask()
        return self.init_state_dict

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
    device = utils.get_device()
    args = utils.get_args()
    logger = utils.setup_logger()
    num_exper = 5

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = args.imp_iter               # 35 was the default
    MODEL_DIR = MODEL_ROOT_DIR + arch_type + "/" + args.dataset + "/"

    # Copying and Saving Initial State
    corr = []

    pruning = LTPruning(model, args.arch, args.prune_perc_per_layer, train_dl,
                        test_dl)

    # Weight Initialization
    model.apply(pruning.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.save_model(model, MODEL_DIR, "/initial_state_dict.pth.tar")

    # Making Initial Mask
    pruning.init_mask()

    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)


    best_accuracy = 0
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros([ITERATION, num_training_epochs], float)
    all_accuracy = np.zeros([ITERATION, num_training_epochs], float)

    # Iterative Magnitude Pruning main loop
    for imp_iter in tqdm(range(ITERATION)):
        best_accuracy = 0
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        if imp_iter != 0:
            pruning.prune_once(initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        logger.debug(f"[{imp_iter + 1}/{ITERATION}] " + "IMP loop. Pruning level "
                     f"{comp[imp_iter]}")

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[imp_iter] = comp_level
        logger.debug(f"Compression level: {comp_level}")

        # Training the network
        for train_iter in tqdm(range(num_training_epochs), leave=False):

            # Training
            logger.debug(f"Training iteration {train_iter} / {num_training_epochs}")
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              epochs=num_epochs, device=device)

            # Test and save the most accurate model
            # logger.debug("Testing...")
            accuracy = test(model, test_dataloader, criterion, device)

            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.save_model(model, MODEL_DIR, f"{imp_iter + 1}_model.pth.tar")

            # apply the controller after some epochs
            if (train_iter == control_at_epoch) and (imp_iter == control_at_iter):
                # network.trained_enough(accuracy, train_dataloader, criterion,
                #                        optimizer, num_epochs, device)
                activations = Activations(model, test_dataloader, device, batch_size)
                # control_corrs.append(activations.get_connectivity())
                control_corrs.append(activations.get_correlations())
                # pruning.apply_controller(control_corrs, [2])
                pruning.controller(control_corrs, activations.layers_dim, control_type, imp_iter, [2])
                # pruning.apply_controller(3, control_corrs)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


            all_loss[imp_iter, train_iter] = loss
            all_accuracy[imp_iter, train_iter] = accuracy


            # Frequency for Printing Accuracy and Loss
            # if train_iter % print_freq == 0:
            #     logger.debug(f'Train Epoch: {train_iter}/{num_training_epochs} \n'
            #                  f'Loss: {loss:.6f} Accuracy: {accuracy:.2f}% \n'
            #                  f'Best Accuracy: {best_accuracy:.2f}%\n')

        # Calculate the connectivity
        # network.trained_enough(accuracy, train_dataloader, criterion, optimizer,
        #                        num_epochs, device) 
        activations = Activations(model, test_dataloader, device, batch_size)
        corr = activations.get_connectivity()
        corrs.append(corr)
        if imp_iter <= control_at_iter:
            control_corrs.append(activations.get_correlations())

        # save the best model
        bestacc[imp_iter] = best_accuracy

        # Dump Plot values
        # all_loss.dump(C.RUN_DIR + f"{prune_type}_all_loss_{comp_level}.dat")
        # all_accuracy.dump(C.RUN_DIR + f"{prune_type}_all_accuracy_{comp_level}.dat")
        pickle.dump(corrs, open(C.RUN_DIR + arch_type + "_correlation.pkl", "wb"))
        pickle.dump(all_accuracy, open(C.RUN_DIR + arch_type + "_all_accuracy.pkl", "wb"))
        pickle.dump(all_loss, open(C.RUN_DIR + arch_type + "_all_loss.pkl", "wb"))

        # Dumping mask
        # with open(C.RUN_DIR + f"{prune_type}_mask_{comp_level}.pkl", 'wb') as fp:
        #     pickle.dump(pruning.mask, fp)

    # Dumping Values for Plotting
    pickle.dump(comp, open(C.RUN_DIR + arch_type + "_compression.pkl", "wb"))
    # bestacc.dump(C.RUN_DIR + f"{prune_type}_bestaccuracy.dat")
    pickle.dump(bestacc, open(C.RUN_DIR + arch_type + "_best_accuracy.pkl", "wb"))
    con_stability = utils.get_stability(corrs)
    print(con_stability)
    pickle.dump(con_stability,
                open(C.RUN_DIR + arch_type + "_connectivity_stability.pkl", "wb"))
    perform_stability = utils.get_stability(all_loss)
    print(perform_stability)
    pickle.dump(perform_stability,
                open(C.RUN_DIR + arch_type + "_performance_stability.pkl", "wb"))
    utils.plot_experiment(bestacc, corrs, C.RUN_DIR + arch_type +
                          "correlation")
    utils.plot_experiment(bestacc, con_stability, C.RUN_DIR + arch_type +
                          "connection-stability")



if __name__ == '__main__':
    main()
