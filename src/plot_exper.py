import plot_tool
import pickle
import utils
import json
from pathlib import Path
import sys
import os
import numpy as np


def making_plots_perf(arch, dataset, layer, exper_cntr, exper_no_cntr):
    print(arch, dataset)
    no_cntr_dir = f"../output/performance/{arch}/{dataset}/no_cntr/" + exper_no_cntr
    cntr_dir = f"../output/performance/{arch}/{dataset}/cntr/{layer}/" + exper_cntr

    labels = [f"{arch} w. $\phi$", f"{arch} w.o. $\phi$"]
    if not os.path.exists(no_cntr_dir):
        print(no_cntr_dir)
        sys.exit(f"Directory for {arch} and {dataset} doesn't exist")

    args = json.load(open(cntr_dir + "exper.json"))

    no_cntr_conn = pickle.load(open(no_cntr_dir + "conn.pkl", "rb"))
    no_cntr_conn = np.mean(no_cntr_conn, axis=0)
    no_cntr_all_acc = pickle.load(open(no_cntr_dir + "all_accuracies.pkl", "rb"))
    no_cntr_all_acc = np.mean(no_cntr_all_acc, axis=0)

    cntr_conn = pickle.load(open(cntr_dir + "conn.pkl", "rb"))
    cntr_conn = np.mean(cntr_conn, axis=0)
    cntr_all_acc = pickle.load(open(cntr_dir + "all_accuracies.pkl", "rb"))
    cntr_all_acc = np.mean(cntr_all_acc, axis=0)

    print(cntr_all_acc)
    print(no_cntr_all_acc)
    # print(cntr_conn)
    # print(no_cntr_conn)

    plot_tool.plot_multi_all_accuracy([no_cntr_all_acc, cntr_all_acc], labels,
                                      args, cntr_dir + "all_accuracies")
    plot_tool.plot_connectivity(cntr_conn, cntr_dir + "conn")
    plot_tool.plot_connectivity(no_cntr_conn, cntr_dir + "no_conn")
    print("plots saved in:", cntr_dir)


def making_plots_eff(arch, dataset):
    no_cntr_dir = f"../output/efficiency/{arch}/{dataset}/no_cntr/"
    cntr_dir = f"../output/efficiency/{arch}/{dataset}/cntr/2/"
    labels = ["w. $\phi$", "w.o. $\phi$"]

    # "/home/soheil/Sync/umaine/bnn/nips/output/test/model/resnet/MNIST/cntr/"
    args = json.load(open(cntr_dir + "exper.json"))
    # print(args['train_epochs'])

    no_cntr_conn = pickle.load(open(no_cntr_dir + "conn.pkl", "rb"))
    no_cntr_all_acc = pickle.load(open(no_cntr_dir + "all_accuracies.pkl", "rb"))

    cntr_conn = pickle.load(open(cntr_dir + "conn.pkl", "rb"))
    cntr_all_acc = pickle.load(open(cntr_dir + "all_accuracies.pkl", "rb"))
    # print([np.where(np.array(exp) > 88.0)[0][0] for exp in cntr_all_acc])
    # print([np.where(np.array(exp) > 88.0)[0][0] for exp in no_cntr_all_acc])
    print([len(exp) for exp in cntr_all_acc])
    print([len(exp) for exp in no_cntr_all_acc])
    # cntr_all_acc = np.array([acc for exp in cntr_all_acc for acc in exp])
    # no_cntr_all_acc = np.array([acc for exp in no_cntr_all_acc for acc in exp])
    print(cntr_all_acc)
    print(no_cntr_all_acc)

    # plot_tool.plot_multi_all_accuracy([no_cntr_all_acc, cntr_all_acc], labels,
    #                                   args, no_cntr_dir + "../" +  "all_accuracies")
    # plot_tool.plot_connectivity(cntr_conn, no_cntr_dir + "../" +  "conn")
    # plot_tool.plot_connectivity(no_cntr_conn, no_cntr_dir + "../" + "no_conn")
    # plot_tool.plot_connectivity(conn)


def main():
    ARCHS=["resnet18", "vgg16"]
    DATASETS=["MNIST", "CIFAR10"]

    exper_cntr = ["", "13-32/", "13-38/", "13-56/", "08-52/"]
    exper_no_cntr = [""]

    layer = 9

    # exper_cntr = ""
    # exper_no_cntr = ""

    making_plots_perf(ARCHS[0], DATASETS[1], layer, exper_cntr[-1],
                      exper_no_cntr[-1])
    # making_plots_efficiency(ARCHS[2], DATASETS[1])
    # for arch in ARCHS:
    #     for dataset in DATASETS:
    #         making_plots(arch, dataset)


    # making_plots_eff(ARCHS[1], DATASETS[1])

if __name__ == '__main__':
    main()
