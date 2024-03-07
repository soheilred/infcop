import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
import sys
import constants as C
# from tueplots import figsizes, fonts
# rc('font',**{'family':'serif','serif':['Times']})
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# plt.rcParams.update(fonts.jmlr2001_tex(family="serif"))
plt.rcParams.update({
    "font.family": "serif",
    # "font.sans-serif": "Liberation Sans",
    # "font.size": 20.0,
    # "font.weight": "bold",
    # "xtick.labelsize": "large",
    # "ytick.labelsize": "large",
    # "legend.loc": "upper right",
    # "axes.labelweight": "bold",
    # "text.usetex": True,
    # "savefig.dpi": 100,     # higher resolution output.
    # "pgf.rcfonts": True,
    # "pgf.texsystem": 'pdflatex', # default is xetex
    # "pgf.preamble": [
    #      r"\usepackage[T1]{fontenc}",
    #      # r"\usepackage{unicode-math}",   # unicode math setup
    #      r"\usepackage{mathpazo}"
    #      ]
})

# matplotlib.use('tkagg')

filled_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p',
                    '*', 'h', 'H', 'D', 'd', 'P', 'X']
linestyles = ['-', '--', '-.', ':']

def plot_all_accuracy(accuracies, filename):
    accuracy_list = accuracies.flatten()
    fig, axs = plt.subplots(1, figsize=(8,4))
    xdata = np.arange(0, 2 * len(accuracy_list) + 0, 2)
    axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Training epochs", ylabel="Accuracy(\%)")

    axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    # axs.set_xticks(xdata)#, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(0, 2 * len(accuracy_list) + 0, 4)
    axs.set_xticks(major_ticks)
    axs.set_xlim([0, 2 * len(accuracy_list)])
    plt.grid()
    plt.savefig(filename + ".png")


def plot_multi_all_accuracy(accuracies, labels, args, filename):
    # accuracies_list = []
    # for i in range(len(accuracies)):
    #     accuracies_list.append(accuracies[i].flatten())
    fig, axs = plt.subplots(1, figsize=(8,4))
    xdata = np.arange(0, args['train_per_epoch'] * args['train_epochs'] *
                      len(accuracies[0]), args['train_per_epoch'])
    axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Training epochs", ylabel="Accuracy(\%)")
    for i in range(len(accuracies)):
        axs.plot(xdata, accuracies[i].flatten(),
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i])
                 # alpha=.5) 

    fig.tight_layout(pad=2.0)
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(0, args['train_per_epoch']* args['train_epochs'] *
                            len(accuracies[0]), 
                            args['train_per_epoch'] * args['train_epochs'])
    axs.set_xticks(major_ticks)
    axs.set_xlim([0, args['train_per_epoch']* args['train_epochs'] * len(accuracies[0])])
    plt.legend()
    plt.grid()
    plt.savefig(filename + ".png")


def plot_experiment(train_acc, ydata, filename):
    fig, axs = plt.subplots(2)
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
    axs[1].set_ylim([-5, 5])
    axs[1].set_title("Correlations between layers")
    axs[1].set(xlabel="Layers", ylabel="Correlation")
    # axs[1].set_xticks(xdata, labels=[str(i) + "-" + str(i + 1) for i in range(len(xdata))])
    axs[1].legend(loc="lower right")
    fig.tight_layout(pad=2.0)
    plt.savefig(filename + ".png")
    # plt.show()


def plot_accuracy():
    # out_dir = "../output/05-02-21-28/"
    out_dir = sys.argv[1]
    # train_acc = pickle.load(open("vgg16" + "_training_acc.pkl", "rb"))
    corrs = pickle.load(open(out_dir + "vgg16" + "_correlation.pkl", "rb"))
    all_accuracy = pickle.load(open(out_dir + "vgg16" + "_all_accuracy.pkl", "rb"))
    max_accuracy = np.max(all_accuracy, axis=1)
    best_accuracy = pickle.load(open(out_dir + "vgg16" + "_best_accuracy.pkl", "rb"))
    all_loss = pickle.load(open(out_dir + "vgg16" + "_all_accuracy.pkl", "rb"))
    # compression = pickle.load(open(out_dir + "vgg16" + "_compression.pkl", "rb"))
    perf_stability = pickle.load(open(out_dir + "vgg16" + "_performance_stability.pkl", "rb")) 
    connect_stability = pickle.load(open(out_dir + "vgg16" + "_connectivity_stability.pkl", "rb")) 

    plot_experiment(best_accuracy, corrs, out_dir + "vgg16_correlation")
    plot_experiment(best_accuracy, connect_stability, out_dir + "vgg16_connectivity_instability")

    # plot_experiment(max_accuracy, corrs, out_dir + "vgg16_correlation")
    # plot_experiment(max_accuracy, connect_stability, out_dir + "vgg16_connectivity_instability")

    plot_all_accuracy(all_accuracy, out_dir + "accuracies")
    print(np.round(corrs, 3))
    print(np.round(connect_stability, 3))


def plot_max_accuracy(accuracies, labels, filename):
    fig, axs = plt.subplots(1, figsize=(5,5))
    xdata = np.arange(1, len(accuracies[0]) + 1)
    # axs.set_title("Accuracy of network in IMP")
    axs.set(xlabel="Iteration", ylabel="Accuracy(\%)")

    for i in range(len(accuracies)):
        axs.plot(xdata, accuracies[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i],
                 alpha=.5) 
    # axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(accuracies[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(accuracies[0])])
    # plt.grid()
    plt.savefig(filename + ".png")
   

def plot_three():
    """Plot the average of three experiments."""
    exper_folder = sys.argv[1:4]
    acc_list = []
    corr_list = []
    for i in range(3):
        acc, corr = get_vars(C.OUTPUT_DIR + exper_folder[i])
        acc_list.append(acc)
        corr_list.append(corr)

    acc_mean = np.mean(np.array(acc_list), axis=0)
    acc_max = np.max(acc_mean, axis=1)
    corr_mean = np.mean(np.array(corr_list), axis=0)
    plot_all_accuracy(acc_mean, C.OUTPUT_DIR + exper_folder[0] + "three_accuracies")
    plot_experiment(acc_max, corr_mean, C.OUTPUT_DIR + exper_folder[0] + "three_vgg16_correlation")
    
def plot_train_epochs(epochs, labels, filename):
    fig, axs = plt.subplots(1, figsize=(5,5))
    xdata = np.arange(1, len(epochs[0]) + 1)
    # axs.set_title("Epochs needed to reach to baseline accuracy")
    axs.set(xlabel="Iteration", ylabel="Epochs")

    for i in range(len(epochs)):
        axs.plot(xdata, epochs[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=labels[i],
                 alpha=.5) 
    # axs.plot(xdata, accuracy_list, 'k')
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(epochs[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(epochs[0])])
    # plt.grid()
    plt.savefig(filename + ".png")

def plot_connectivity(conns, filename):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    xdata = np.arange(1, len(conns[0]) + 1)

    for i in range(len(conns)):
        axs.plot(xdata, conns[i], marker=filled_markers[i],
                 linestyle=linestyles[i % len(linestyles)],
                 label=f"Iter {i}",
                 alpha=.5) 
    fig.tight_layout(pad=2.0)
    plt.legend()
    # axs.set_xticks(xdata, labels=[i for i in range(0, 2 * len(xdata), 20)])
    major_ticks = np.arange(1, len(conns[0]) + 1)
    axs.set_xticks(major_ticks)
    # axs.set_xlim([1, len(epochs[0])])
    axs.set_title("Accuracy of network in training")
    axs.set(xlabel="Layer index", ylabel="Correlations")
    plt.grid()
    plt.savefig(filename + ".png")
 

def plot_correlations(filename):
    corrs = pickle.load(open(filename, "rb"))

    import ipdb; ipdb.set_trace()

def main():
    # plot_accuracy()
    plot_correlations(sys.argv[1])
    
if __name__ == '__main__':
    main()
