import pickle
import utils
import numpy as np
# import matplotlib.pyplot as plt


out_dir = "../output/05-02-15-35/"
# train_acc = pickle.load(open("vgg16" + "_training_acc.pkl", "rb"))
corr = pickle.load(open(out_dir + "vgg16" + "_correlation.pkl", "rb"))
all_accuracy = pickle.load(open(out_dir + "vgg16" + "_all_accuracy.pkl", "rb"))
all_accuracy_max = all_accuracy
# all_accuracy_max = np.max(all_accuracy, axis=1)
compression = pickle.load(open(out_dir + "vgg16" + "_compression.dat", "rb"))
perf_stability = pickle.load(open(out_dir + "performance_stability.pkl", "rb")) 
connect_stability = pickle.load(open(out_dir + "connectivity_stability.pkl", "rb")) 

utils.plot_experiment(all_accuracy_max, connect_stability, out_dir + "vgg16_connectivity_stability")

