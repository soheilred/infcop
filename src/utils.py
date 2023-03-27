import os
import torch
import pickle
from torch import nn
import numpy as np
import logging
import constants as C
import matplotlib.pyplot as plt
from pathlib import Path

log = logging.getLogger("sampleLogger")


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, directory, name):
    checkdir(directory)
    torch.save(model, directory + name)

def batch_mul(mat, v):
    in_shape = mat.shape[:2]
    res = torch.stack([v[i, j] * mat[i, j, :, :] for i in range(in_shape[0]) for j in range(in_shape[1])])
    return res.reshape(mat.shape)

def get_vars(arch, out_dir):
    # train_acc = pickle.load(open("vgg16" + "_training_acc.pkl", "rb"))
    corrs = pickle.load(open(out_dir + arch + "_correlation.pkl", "rb"))
    all_accuracy = pickle.load(open(out_dir + arch + "_all_accuracy.pkl", "rb"))
    # max_accuracy = np.max(all_accuracy, axis=1)
    best_accuracy = pickle.load(open(out_dir + arch + "_best_accuracy.pkl", "rb"))
    all_loss = pickle.load(open(out_dir + arch + "_all_loss.pkl", "rb"))
    # compression = pickle.load(open(out_dir + "vgg16" + "_compression.pkl", "rb"))
    # perf_stability = pickle.load(open(out_dir + "vgg16" + "_performance_stability.pkl", "rb")) 
    # connect_stability = pickle.load(open(out_dir + "vgg16" + "_connectivity_stability.pkl", "rb")) 
    return all_accuracy, corrs

def get_mean_train_epochs(arch, out_dir):
    epochs = []
    for odir in out_dir:
        epochs.append(pickle.load(open(C.OUTPUT_DIR + odir + arch + "_train_epochs.pkl", "rb")))
    print(np.sum(epochs, axis=1))
    return np.mean(epochs, axis=0)

def get_max_accuracy(arch, exper_dirs):
    acc_list = []
    corr_list = []
    for i in range(len(exper_dirs)):
        acc, corr = get_vars(arch, C.OUTPUT_DIR + exper_dirs[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    acc_max = np.max(acc_mean, axis=1)
    # corr_mean = np.mean(np.array(corr_list), axis=0)
    return acc_max

def get_mean_accuracy(arch, exper_dirs):
    acc_list = []
    corr_list = []
    for i in range(len(exper_dirs)):
        acc, corr = get_vars(arch, C.OUTPUT_DIR + exper_dirs[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    print(np.mean(acc_list, axis=(1, 0)))
    return acc_mean


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero / total) * 100, 1))

def setup_logger():
    import logging
    import logging.config

    Path(C.RUN_DIR).mkdir(parents=True, exist_ok=True)
    logging.config.fileConfig(C.LOG_CONFIG_DIR,
                              defaults={'logfilename': C.LOG_FILENAME})
    # create logger
    logger = logging.getLogger("sampleLogger")
    # 'application' code
    logger.debug("In " + os.uname()[1])
    return logger

def plot_pruning():
    # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
    # NOTE Loss is computed for every iteration while Accuracy is computed
    # only for every {args.valid_freq} iterations. Therefore Accuracy saved
    # is constant during the uncomputed iterations.
    # NOTE Normalized the accuracy to [0,100] for ease of plotting.
    plt.plot(np.arange(1,(args.num_training_epochs)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss")
    plt.plot(np.arange(1,(args.num_training_epochs)+1), all_accuracy, c="red", label="Accuracy")
    # plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss and Accuracy")
    # plt.legend()
    # plt.grid(color="gray")
    # plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp_level}.png", dpi=1200) 
    # plt.close()
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

def get_stability(in_measure):
    in_measure = np.array(in_measure)
    stability = [np.divide(in_measure[i] - in_measure[i + 1],
                           in_measure[i]) for i in range(in_measure.shape[0] - 1)]
    return stability

def plot_experiment(train_acc, ydata, filename):
    import matplotlib
    matplotlib.use('tkagg')

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
    axs[0].set(xlabel="IMP Iteration", ylabel="Training accuracy")
    axs[0].set_xticks(xdata)
    # axs[0].legend(loc="upper right")
    xdata = np.arange(discard, len(ydata[0]))
    for i in range(len(ydata)):
        axs[1].plot(xdata, ydata[i][discard:],
                       # marker=(5, i),
                       marker=filled_markers[i],
                       label="Itr. " + str(i),
                       alpha=.5)
        # axs[1].set_yscale('log')
    # axs.set_xlim([0, 1])
    # axs.set_ylim([0, 1])
    axs[1].set_title("Correlations between layers")
    axs[1].set(xlabel="Layers", ylabel="Correlation")
    # axs[1].set_xticks(xdata, labels=[str(i) + "-" + str(i + 1) for i in range(len(xdata))])
    axs[1].legend(loc="lower right")
    fig.tight_layout(pad=2.0)
    plt.savefig(filename + ".png")
    # plt.show()
