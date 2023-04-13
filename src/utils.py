import os
import torch
import pickle
import argparse
import json
from torch import nn
import numpy as np
import logging
import constants as C
import matplotlib.pyplot as plt
from pathlib import Path
# from varname import nameof

import logging
import logging.config
logger = logging.getLogger("sampleLogger")


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

def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def save_vars(**variables):
    for varname, value in variables.items():
        # print(nameof(var))
        pickle.dump(value, open(C.OUTPUT_DIR + # arch + "_" + 
                                    varname + ".pkl", "wb"))
    
def load_checkpoints(args):
    save_prefix = C.CHPT_DIR + str(args.dataset) +\
                str(args.prune_perc_per_layer) + str(args.run_id) +\
                str(args.task_num) 
    previoustaskpath = C.CHPT_DIR + str(args.dataset) +\
                        str(args.prune_perc_per_layer) + str(args.run_id) +\
                        str(args.task_num-1) 
    os.makedirs(save_prefix, exist_ok = True)
    os.makedirs(previoustaskpath, exist_ok = True)

    trainedpath = os.path.join(save_prefix, "trained.pt")
    initpath = os.path.join(previoustaskpath, "final.pt")

    if os.path.isfile(initpath) == False and args.task_num == 0:
        print("initializing model",flush = True)
        utils.init_dump(args.arch, initpath)
    
    if os.path.isfile(os.path.join(previoustaskpath,"final.pt")) == True and (args.mode == "t" or args.mode == "all"):
        ckpt = torch.load(os.path.join(previoustaskpath,"final.pt"))

    elif os.path.isfile(os.path.join("../checkpoints/",str(args.dataset),str(args.prune_perc_per_layer),str(args.run_id),str((len(taskset)-1)), "final.pt")) == True and args.mode == "e":
        ckpt = torch.load(os.path.join("../checkpoints/", str(args.dataset), str(args.prune_perc_per_layer), str(args.run_id), str((len(taskset)-1)), "final.pt"))

    elif os.path.isfile(trainedpath) == True and (args.mode == "p" or args.mode == "c"):
        ckpt = torch.load(trainedpath)

    else:
        print("No checkpoint file found")
        return 0

    
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using {device} device")
    if torch.cuda.is_available():
        logger.debug("Name of the Cuda Device: " +
                     torch.cuda.get_device_name())
    return device


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
    # # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    return (round((nonzero / total) * 100, 1))

def setup_logger_dir():
    Path(C.RUN_DIR).mkdir(parents=True, exist_ok=True)
    logging.config.fileConfig(C.LOG_CONFIG_DIR,
                              defaults={'logfilename': C.LOG_FILENAME})
    return setup_logger()

def setup_logger():
    logging.config.fileConfig(C.LOG_CONFIG_DIR)
    logger = logging.getLogger("sampleLogger")
    logger.debug("In " + os.uname()[1])
    return logger

def get_stability(in_measure):
    in_measure = np.array(in_measure)
    stability = [np.divide(in_measure[i] - in_measure[i + 1],
                           in_measure[i]) for i in range(in_measure.shape[0] - 1)]
    return stability

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        choices=['vgg11', 'vgg16', 'resnet18', 'densenet121'],
                        default='resnet',
                        help='Architectures')
    parser.add_argument('--mode', choices=['t','c','p','e','all'], default='all',
                        help='modes: train, calc. conns, prune, finetune, or eval')
    parser.add_argument('--pretrained', type=str, default="True",
                        choices=["False","True"],
                        help='Start with a pretrained network?')
    parser.add_argument('--dataset', type=str,
                        choices=['CIFAR10', 'MNIST', 'pmnist', '6splitcifar', '11splitcifar'],
                        default='CIFAR10', help='Name of dataset')
    parser.add_argument('--single_task', action='store_true',
                        default=False, help='Run only the current task')
    parser.add_argument('--task_num', type=int, default=0,
                        help='Current task number.')
    parser.add_argument('--run_id', type=str, default="000",
                        help='Id of current run.')
    parser.add_argument('--num_outputs', type=int, default=-1,
                        help='Num outputs for dataset')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of CPU cores.')

    parser.add_argument('--save_prefix', type=str, default='../checkpoints/',
                      help='Location to save model')
    parser.add_argument('--loadname', type=str, default='',
                      help='Location to save model')
    # Training options.
    parser.add_argument('--train_epochs', type=int, default=2,
                      help='Number of epochs to train for')

    parser.add_argument('--train_per_epoch', type=int, default=2,
                      help='Number of epochs to train for')

    parser.add_argument('--lr', type=float, default=0.1,
                      help='Learning rate')

    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                      help='Weight decay')

    parser.add_argument('--Milestones', nargs='+', type=float,
                        default=[30,60,90])
    
    parser.add_argument('--Gamma', type=float, default=0.1)   
    # Pruning options.
    parser.add_argument('--prune_method', type=str, default='sparse',
                      choices=['sparse'], help='Pruning method to use')
    parser.add_argument('--prune_perc_per_layer', type=float, default=0.1,
                      help='% of neurons to prune per layer')
    parser.add_argument('--finetune_epochs', type=int, default=2,
                      help='Number of epochs to finetune for after pruning')
    parser.add_argument('--freeze_perc', type=float, default=0.0)                   
    parser.add_argument('--num_freeze_layers', type=int, default=0)     
    parser.add_argument('--freeze_order', choices=['top','bottom', 'random'],
                      default=['top'],
                      help='Order of selection for layer freezing, by connectivity')
    # controller
    parser.add_argument('--control-at-iter', default=-1,
                      help='Iteration at which the controller is applied')

    parser.add_argument('--control_at_epoch', default=2,
                      help='Epoch at which the controller is applied')

    parser.add_argument('--acc_thrd', default=70,
                      help='Threshold accuracy to stop the training loop')

    parser.add_argument('--control_type', default=1,
                      help='1: correlation, 2: connectivity, 3: prev weights')

    parser.add_argument('--imp_total_iter', default=10,
                      help='Number of iteration at IMP')
    args = parser.parse_args()

    # this line will be changed. The address of the file should match with the
    # experiment 
    json.dump(args.__dict__, open("exper.json", 'w'), indent=2)
    return args


