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
import plot_tool
from data_loader import Data
from network import Network, train, test
import constants as C

log = logging.getLogger("sampleLogger")
log.debug("In " + os.uname()[1])


class Activations:
    def __init__(self, model, dataloader, device, batch_size):
        """
        Parameters
        ----------
        model: Network
            The network we want to calculate the connectivity for.
        dataloader: Data
            The data we are drawing from, usuaully the test set.
        device: string
            "cpu" or "cuda"
        batch_size: int
            Batch size
        """
        self.activation = {}
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.batch_size = batch_size
        self.layers_dim = None
        self.layers_idx = None
        self.act_keys = None

    def hook_fn(self, m, i, o):
        """Assign the activations/mean of activations to a matrix
        Parameters
        ----------
        self: type
            description
        m: str
            Layer's name
        i: type
            description
        o: torch tensor
            the activation function output
        """

        tmp = o.detach()
        if len(tmp.shape) > 2:
            self.activation[m] = torch.mean(tmp, axis=(2, 3))
            # activation[m] = tmp
        else:
            self.activation[m] = tmp

    def hook_all_layers(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)

    def set_layers_idx(self):
        """Find all convolutional and linear layers and add them to layers_ind.
        """
        layers_idx = []
        layers_dim = []
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                layers_idx.append(module_idx)
                layers_dim.append(module[1].weight.shape)

        self.layers_dim = layers_dim
        self.layers_idx = layers_idx

    def get_layers_idx(self):
        if self.layers_idx == None:
            self.set_layers_idx()
        return self.layers_idx

    def hook_layer_idx(self, item_key, hook_handles):
        for module_idx, module in enumerate(self.model.named_modules()):
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                if (module_idx == item_key):
                    hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                    self.layers_dim.append(module[1].weight.shape)

    def set_act_keys(self):
        layers_dim = []
        hook_handles = []

        self.hook_all_layers(layers_dim, hook_handles)

        with torch.no_grad():
            X, y = next(iter(self.dataloader))
            X, y = X.to(self.device), y.to(self.device)
            self.model(X)
            act_keys = list(self.activation.keys())

        self.act_keys = act_keys

    def get_act_keys(self):
        if self.act_keys == None:
            self.set_act_keys()
        return self.act_keys

    def get_corrs(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        corrs = []

        for idx in range(num_layers - 1):
            logging.debug(f"working on layer {layers_idx[idx]} {str(act_keys[idx])[:18]}...")
            # prepare an array with the right dimension
            parent_arr = []
            child_arr = []

            # add the sample's activation to the array

            with torch.no_grad():
                for batch, (X, y) in enumerate(self.dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    self.model(X)
                    parent_arr.append(self.activation[act_keys[idx]].\
                                  detach().cpu().numpy())
                    child_arr.append(self.activation[act_keys[idx + 1]].\
                                 detach().cpu().numpy())

            parent = np.vstack(parent_arr)
            parent = (parent - parent.mean(axis=0))
            parent /= np.abs(np.max(parent))
            child = np.vstack(child_arr)
            child = (child - child.mean(axis=0))
            child /= np.abs(np.max(child)) # child.std(axis=0)
            if np.any(np.isnan(parent)):
                print("nan in layer {layers_idx[idx]}")
                import ipdb; ipdb.set_trace()

            if np.any(np.isnan(child)):
                print("nan in layer {layers_idx[idx + 1]}")
                import ipdb; ipdb.set_trace()
            # corr = np.corrcoef(parent, child, rowvar=False)
            # x_len = corr.shape[0] // 2
            # y_len = corr.shape[1] // 2
            corr = utils.batch_mul(parent, child)
            logging.debug(f"correlation dimension: {corr.shape}, conn: {np.mean(corr):.6f}")
            corrs.append(corr)

        # print(corrs)
        self.model.train()
        return corrs

    def get_conns(self, corrs):
        conns = []
        for corr in corrs:
            conns.append(corr.mean())
        print(conns)
        return conns

    def get_act_layer(self, layers_dim, hook_handles):
        """ Hook a handle to all layers that are interesting to us, such as
        Linear or Conv2d.
        Parameters
        ----------
        net: Network
            The network we're looking at
        layers_dim: list
            List of layers that are registered for saving activations
        hook_handles: type
            description

        If it is a sequential, don't register a hook on it but recursively
        register hook on all it's module children
        """
        for module in self.model.named_modules():
            if isinstance(module[1], nn.Conv2d) or \
                         isinstance(module[1], nn.Linear):
                hook_handles.append(module[1].register_forward_hook(self.hook_fn))
                self.layers_dim.append(module[1].weight.shape)

        # Recursive version
        # for name, layer in net._modules.items():
        #     if (isinstance(layer, nn.Sequential) or
        #             isinstance(layer, ResidualBlock) or
        #             isinstance(layer, Bottleneck)):
        #         self.get_all_layers(layer, layers_dim, hook_handles)
        #     elif (isinstance(layer, nn.Conv2d) or
        #           (isinstance(layer, nn.Linear))):
        #         # it's a non sequential. Register a hook
        #         hook_handles.append(layer.register_forward_hook(self.hook_fn))
        #         layers_dim.append(layer.weight.shape)

    def get_correlations(self):
        """ Compute the individual correlation
        Returns
        -------
        List of 2d tensors, each representing the connectivity between two
        consecutive layer.
        """
        self.model.eval()
        ds_size = len(self.dataloader.dataset)
        num_batch = len(self.dataloader)
        # params = list(self.model.parameters())

        layers_idx = self.get_layers_idx()
        layers_dim = self.layers_dim
        num_layers = len(layers_dim)
        act_keys = self.get_act_keys()
        import ipdb; ipdb.set_trace()

        corrs = [torch.zeros((layers_dim[i][0], layers_dim[i + 1][0])).to(self.device)
                 for i in range(num_layers - 1)]

        act_means = [torch.zeros(layers_dim[i][0]).to(self.device)
                            for i in range(num_layers)]
        act_sq_sum = [torch.zeros(layers_dim[i][0]).to(self.device)
                         for i in range(num_layers)]
        act_max = torch.zeros(num_layers).to(self.device)

        with torch.no_grad():
            # Compute the mean of activations
            log.debug("Compute the mean and sd of activations")
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                self.model(X)
                for i in range(num_layers):
                    act_means[i] += torch.sum(self.activation[act_keys[i]],dim=0)
                    act_sq_sum[i] += torch.sum(
                            torch.pow(self.activation[act_keys[i]], 2), dim=0)
                    act_max[i] = torch.max(self.activation[act_keys[i]])

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
                    f0 = torch.div((self.activation[act_keys[i]] -
                                    act_means[i]), act_max[i]).T
                    f1 = torch.div((self.activation[act_keys[i + 1]] -
                                    act_means[i + 1]), act_max[i + 1])
                    corrs[i] += torch.matmul(f0, f1)

        self.model.train()
        return corrs

    def get_connectivity(self):
        """Find the connectivity of each layer, the mean of correlation matrix.
        """
        self.model.eval()
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
        act_max = [torch.zeros(layers_dim[i][0]).to(self.device)
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
                    act_max[i] = torch.max(self.activation[act_keys[i]])
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
                    f0 = torch.div((self.activation[act_keys[i]] -
                                    activation_means[i]), act_max[i]).T
                    f1 = torch.div((self.activation[act_keys[i + 1]] -
                                    activation_means[i + 1]), act_max[i + 1])

                    # Zero-meaned activations
                    # f0 = (self.activation[act_keys[i]] - activation_means[i]).T
                    # f1 = (self.activation[act_keys[i + 1]] - activation_means[i + 1])

                    corrs[i] += torch.matmul(f0, f1)
        # Remove all hook handles
        for handle in hook_handles:
            handle.remove()

        self.model.train()
        return [torch.mean(corrs[i]).item()/(layers_dim[i][0] * layers_dim[i + 1][0])
                for i in range(num_layers - 1)]


def main():
    # preparing the hardware
    args = utils.get_args()
    device = utils.get_device(args)
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
    test_acc = torch.zeros(num_exper)

    for i in range(num_exper):
        logger.debug("=" * 10 + " experiment " + str(i + 1) + "=" * 10)
        train_acc, _ = train(model, train_dl, loss_fn, optimizer,
                             args.train_epochs, device)
        test_acc[i] = test(model, test_dl, loss_fn, device)

        activations = Activations(model, test_dl, device, args.batch_size)
        # corr.append(activations.get_connectivity())
        corrs = activations.get_corrs()
        my_corrs = activations.get_correlations()
        import ipdb; ipdb.set_trace()

        diff = [torch.sum(corrs[i] - my_corrs[i]) for i in range(len(corrs))]
        corr.append(corrs)

        utils.save_model(model, C.OUTPUT_DIR, args.arch + f'-{i}-model.pt')
        logger.debug('model is saved...!')

        utils.save_vars(test_acc=test_acc, corr=corr)

    print(corr)
    plot_tool.plot_connectivity(test_acc, corr)


if __name__ == '__main__':
    main()
