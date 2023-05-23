import plot_tool
import pickle
import utils
import json
from pathlib import Path
import sys
import os
import numpy as np


def making_plots_performance(arch, dataset, exper_cntr, exper_no_cntr):
    print(arch, dataset)
    no_cntr_dir = f"../output/performance/{arch}/{dataset}/no_cntr/" + exper_no_cntr
    cntr_dir = f"../output/performance/{arch}/{dataset}/cntr/2/" + exper_cntr

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
                                      args, no_cntr_dir + "../" + "all_accuracies")
    plot_tool.plot_connectivity(cntr_conn, no_cntr_dir + "../" +  "conn")
    plot_tool.plot_connectivity(no_cntr_conn, no_cntr_dir + "../" + "no_conn")


def making_plots_efficiency(arch, dataset):
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

    exper_cntr = ["09-05-00-06-59/", "09-05-13-06-08/", "09-05-21-22-08/",
                "09-05-13-15-53/", "09-05-12-39-57/", "09-06/", "09-22/",
                    "08-10/", "16-13/", "21-18/", "21-21/" , "09-05-14-13-10/",
                  "09-05-13-06-08/"]
    exper_no_cntr = ["09-05-00-07-49/", "09-05-12-42-23/", "09-05-21-22-08/",
                    "09-05-13-11-53/", "09-05-12-36-16/", "09-06/", "09-22/",
                     "08-10/", "23-57/", "21-18/", "21-21/", "09-05-12-44-51/",
                     "09-05-12-42-23/"]

    # exper_cntr = ""
    # exper_no_cntr = ""

    making_plots_performance(ARCHS[0], DATASETS[1], exper_cntr[12],
                             exper_no_cntr[12])
    # making_plots_efficiency(ARCHS[2], DATASETS[1])
    # for arch in ARCHS:
    #     for dataset in DATASETS:
    #         making_plots(arch, dataset)


    making_plots_efficiency(ARCHS[1], DATASETS[1])

if __name__ == '__main__':
    main()
