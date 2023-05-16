import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms

# import logging
# import constants as C
import clmodels

# log = logging.getLogger("sampleLogger")

class Network():
    def __init__(self, args, pretrained="True"):
        """Setup the network and adjust it to the dataset dimentions.
        Parameters
        ----------
        device: str
            "cpu" or "cuda"
        pretrained: bool
            want the network to be pretrained or not
        """
        self.args = args
        self.arch = args.arch
        self.cuda = args.cuda

        self.preprocess = None
        self.model = None
        self.pretrained = pretrained
        self.input_size = 32
        self.datasets, self.classifiers = [], nn.ModuleList()

        if self.arch == "resnet18":
                self.model = clmodels.resnet18()
        else:
            sys.exit("Wrong architecture")

        if self.cuda:
            self.model = self.model.cuda()
    
    

    """
    The Network class is responsible for low-level functions which manipulate the model, such as training, evaluating, or selecting the classifier layer
    """
    
    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_classes))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.model.classifier = self.classifiers[self.datasets.index(dataset)]


