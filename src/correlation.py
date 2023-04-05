import os
import sys
import datetime
import torch
import pickle
import numpy as np
import pathlib
import logging
import logging.config
from torch import nn
from models import ResidualBlock
from torchvision.models.resnet import Bottleneck

import utils
from data_loader import Data
from network import Network, train, test
import constants as C

log = logging.getLogger("sampleLogger")
log.debug("In " + os.uname()[1])


class Activations:
    def __init__(self, model, dataloader, device, batch_size):
        "docstring"
        self.activation = {}
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.batch_size = batch_size
        self.layers_dim = []

    def hook_fn(self, m, i, o):
        tmp = o.detach()
        if len(tmp.shape) > 2:
            self.activation[m] = torch.mean(tmp, axis=(2, 3))
            # activation[m] = tmp
        else:
            self.activation[m] = tmp

    def get_all_layers(self, net, layers_dim, hook_handles):
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        for name, layer in net._modules.items():
            if (isinstance(layer, nn.Sequential) or
                    isinstance(layer, ResidualBlock) or
                    isinstance(layer, Bottleneck)):
                self.get_all_layers(layer, layers_dim, hook_handles)
            elif (isinstance(layer, nn.Conv2d) or
                  (isinstance(layer, nn.Linear))):
                # it's a non sequential. Register a hook
                hook_handles.append(layer.register_forward_hook(self.hook_fn))
                layers_dim.append(layer.weight.shape)


    def get_correlations(self):
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_dim = self.layers_dim
        hook_handles = []
        self.get_all_layers(self.model, layers_dim, hook_handles)
        num_layers = len(layers_dim)
        first_run = 1
        torch.set_printoptions(precision=4)

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).to(self.device)
                 for i in range(num_layers - 1)]


        act_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        act_sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]
        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                if first_run:
                    act_keys = list(self.activation.keys())
                    first_run = 0

                for i in range(num_layers):
                    act_means[i] += torch.sum(
                        self.activation[act_keys[i]], dim=0)
                    act_sq_sum[i] += torch.sum(
                        torch.pow(self.activation[act_keys[i]], 2), dim=0)

            act_means = [act_means[i] / ds_size for i in range(num_layers)]
            activation_sd = [torch.pow(act_sq_sum[i] / ds_size -
                                       torch.pow(act_means[i], 2), 0.5)
                             for i in range(num_layers)]

            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                for i in range(num_layers - 1):
                    f0 = (self.activation[act_keys[i]] - act_means[i]).T
                    f1 = (self.activation[act_keys[i + 1]] -
                          act_means[i + 1])
                    corrs[i] += torch.matmul(f0, f1)

        return corrs

    def get_connectivity(self):
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_dim = []
        hook_handles = []
        self.get_all_layers(self.model, layers_dim, hook_handles)
        num_layers = len(layers_dim)
        first_run = 1
        torch.set_printoptions(precision=4)

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).to(self.device)
                 for i in range(num_layers - 1)]
        activation_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]

        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                if first_run:
                    act_keys = list(self.activation.keys())
                    first_run = 0

                for i in range(num_layers):
                    activation_means[i] += torch.sum(
                        self.activation[act_keys[i]], dim=0)
                    sq_sum[i] += torch.sum(
                        torch.pow(self.activation[act_keys[i]], 2), dim=0)
            # log.debug("-------------------------------")

            activation_means = [activation_means[i] / ds_size
                                for i in range(num_layers)]
            activation_sd = [torch.pow(sq_sum[i] / ds_size -
                                       torch.pow(activation_means[i], 2), 0.5)
                             for i in range(num_layers)]

            # Compute normalized correlation
            log.debug("Compute normalized correlation")
            for batch, (X, y) in enumerate(self.dataloader):
                # if batch % 100 == 0:
                    # log.debug(f"batch [{batch}/{num_batch}]")

                X, y = X.to(self.device), y.to(self.device)
                self.model(X)

                # if first_run:
                #     act_keys = list(self.activation.keys())
                #     first_run = 0
                for i in range(num_layers - 1):
                    # Normalized activations
                    # f0 = torch.div((self.activation[act_keys[i]] -
                    #                 activation_means[i]), activation_sd[i]).T
                    # f1 = torch.div((self.activation[act_keys[i + 1]] -
                    #                 activation_means[i + 1]), activation_sd[i + 1])

                    # Zero-meaned activations
                    f0 = (self.activation[act_keys[i]] - activation_means[i]).T
                    f1 = (self.activation[act_keys[i + 1]] - activation_means[i + 1])

                    corrs[i] += torch.matmul(f0, f1)

                    # corrs[i] += torch.matmul(torch.flatten(self.activation[act_keys[i]],
                    #                                        start_dim=1).T,
                    #                          torch.flatten(self.activation[act_keys[i + 1]],
                    #                                        start_dim=1))

            # for i in range(num_layers - 1):
            #     if (torch.any(corrs[i] > 10e5)) or (torch.any(corrs[i] < -10e5)):
            #         log.debug(f"Error in layer {i} in correlation")
            #         log.debug(f"{torch.mean(corrs[i])}, {layers_dim[i]}")
        # if w == 0:
            # import ipdb; ipdb.set_trace()
            # log.debug("-------------------------------")
            #     corrs[i] += torch.matmul(torch.flatten(activation[str(i+1)],
            #                                            start_dim=1).T,
            #                              torch.flatten(activation[str(i+2)],
            #                                            start_dim=1))
        # if w == 0:
        # weight_list = [torch.ones(corrs[i].shape).to(device) for i in range(num_layers)]
        # if w == 1:
        #     weight = model.fc1.weight.cpu().detach().T
        # corrs_out = [torch.mean(
        #             torch.mul(corrs[i] / ds_size, weight_list[i])).item()
        #             for i in range(num_layers)]
        # adj_corr = torch.mul(corrs[0], weight)
        # return torch.mean(adj_corr).item()

        # Remove all hook handles
        for handle in hook_handles:
            handle.remove()

        return [torch.mean(corrs[i]).item()/(layers_dim[i][0] * layers_dim[i + 1][0])
                for i in range(num_layers - 1)]



    def get_np_correlation(self):
        corrs = np.zeros(len(self.layers_dim))



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


    corr = []
    train_acc = torch.zeros(num_exper)

    for i in range(num_exper):
        logger.debug("=" * 10 + " experiment " + str(i + 1) + "=" * 10)
        train_acc[i], _ = train(model, train_dl, loss_fn, optimizer,
                             args.train_epochs, device)

        activations = Activations(model, test_dl, device, args.batch_size)
        corr.append(activations.get_correlations())

        utils.save_model(model, C.OUTPUT_DIR, args.arch + str(i) + '-model.pt')
        logger.debug('model is saved...!')

        utils.save_vars(train_acc=train_acc, corr=corr)

    plot_experiment(train_acc, corr, arch)


if __name__ == '__main__':
    main()
