import os
import sys
import datetime
import torch
import pickle
import numpy as np
import pathlib
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('tkagg')
# from matplotlib import rc
# import matplotlib.font_manager as fm
from torch import nn
from data_loader import Data
# from torch.utils.data.sampler import SubsetRandomSampler
# import torch.nn.functional as F
# from torchvision.transforms import ToTensor
# import torchvision.models as models
from network import Network, train, test

# from utils import Activations
from models import ResidualBlock
from torchvision.models.resnet import Bottleneck

import logging
import logging.config
# from EDGE_4_4_1 import EDGE
# torch.cuda.empty_cache()
# from matplotlib.ticker import MaxNLocator
# plt.style.use('ggplot')
# rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
# rc('text', usetex=True)


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

    def get_np_correlation(self):
        corrs = np.zeros(len(self.layers_dim))



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




def plot_experiment(train_acc, ydata, arch):
    # plt.rcParams["font.family"] = "sans-serif"
    # fig, axs = plt.subplots(2, sharex=True)
    # print(fm.get_font_names())
    # print(fm.fontManager.findfont(fontext='ttf'))
    # for f in fm.fontManager.get_font_names():
    #     print(f)
    filled_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p',
                      '*', 'h', 'H', 'D', 'd', 'P', 'X']
    fig, axs = plt.subplots(2)
    # axs.plot(xdata, color='blue')
    discard = 0
    xdata = np.arange(len(train_acc))
    axs[0].scatter(xdata, train_acc, marker=(5, 0))
    axs[0].set_title("Accuracy of network in training")
    axs[0].set(xlabel="Experiment number", ylabel="Training accuracy")
    axs[0].set_xticks(xdata)
    # axs[0].legend(loc="upper right")
    xdata = np.arange(discard, len(ydata[0]))
    for i in range(len(ydata)):
        axs[1].plot(xdata, ydata[i][discard:],
                       # marker=(5, i),
                       marker=filled_markers[i],
                       label="exp. " + str(i),
                       alpha=.5)
        # axs[1].set_yscale('log')
    # axs.set_xlim([0, 1])
    # axs.set_ylim([0, 1])
    axs[1].set_title("Correlations between layers")
    axs[1].set(xlabel="Layers", ylabel="Correlation")
    # axs[1].set_xticks(xdata, labels=[str(i) + "-" + str(i + 1) for i in range(len(xdata))])
    axs[1].legend(loc="lower right")
    fig.tight_layout(pad=2.0)
    plt.savefig(OUTPUT_DIR + arch + "-correlation.png")
    # plt.show()


def main():
    # preparing the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())

    # setting hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 10

    # arch = "vgg11"
    arch = "vgg16"
    # arch = "resnet"
    # arch = "alexnet"
    pretrained = True
    # pretrained = False

    network = Network(device, arch, pretrained)
    preprocess = network.preprocess
    data = Data(batch_size, DATA_DIR, transform=preprocess)
    train_dataloader, test_dataloader = data.train_dataloader, data.test_dataloader
    num_exper = 5
    corr = []
    train_acc = torch.zeros(num_exper)
    mode = sys.argv[1]
    torch.set_printoptions(precision=4)

    for i in range(num_exper):
        torch.manual_seed(5 * i)
        logger.debug("=" * 10 + " experiment " + str(i + 1) + "=" * 10)
        if mode == "train":
            logger.debug('Training in correlation code for a ' + arch)
            model = network.set_model()

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            train_acc[i] = train(model, train_dataloader, loss_fn, optimizer,
                                 num_epochs + i * 4, device)
            activations = Activations(model, test_dataloader, device, batch_size)
            corr.append(activations.get_correlation())
            torch.save(model, MODEL_DIR + arch + str(i) + '-model.pt')
            logger.debug('model is saved...!')

        elif mode == "correlation":
            logger.debug('Calculating correlation')
            model = torch.load(MODEL_DIR + arch + str(i) + '-model.pt')
            model.to(device)
            model.eval()
            activations = Activations(model, test_dataloader, device, batch_size)
            corr.append(activations.get_correlation())
            # corr_w1.append(correlation(model, test_dataloader, device, w=1))

        elif mode == "plot":
            continue

        else:
            sys.exit("Wrong flag")

    if mode == "train":
        pickle.dump(train_acc, open(OUTPUT_DIR + arch + "_training_acc.pkl", "wb"))
        pickle.dump(corr, open(OUTPUT_DIR + arch + "_correlation.pkl", "wb"))

    elif mode == "correlation":
        train_acc = pickle.load(open(OUTPUT_DIR + arch + "_training_acc.pkl", "rb"))
        logger.debug(len(corr))
        pickle.dump(corr, open(OUTPUT_DIR + arch + "_correlation.pkl", "wb"))

    elif mode == "plot":
        train_acc = pickle.load(open(OUTPUT_DIR + arch + "_training_acc.pkl", "rb"))
        corr = pickle.load(open(OUTPUT_DIR + arch + "_correlation.pkl", "rb"))

    plot_experiment(train_acc, corr, arch)

    logger.debug(train_acc)
    logger.debug(corr)

    # if os.path.exists(MODEL_DIR + "model.pt"):
    #     model = torch.jit.load(MODEL_DIR + "model.pt")
    #     model.eval()
    # torch.save(model.state_dict(), MODEL_DIR)
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(MODEL_DIR + 'model.pt') # Save
    # layer = [model.linear_stack[2 * i].weight.cpu().detach().numpy() for i in range(5)]
    # for i in range(5 - 1):
    #     print("Computing I(X, Y) between layers", i, "and", i+1)
    #     I[i] = EDGE(layer[i].flatten(), layer[i+1].flatten())
    # print(I)


if __name__ == '__main__':
    main()
