import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet18_Weights, ResNet50_Weights,\
    vgg11, vgg16, alexnet, VGG11_Weights, VGG16_Weights, AlexNet_Weights

import logging
import utils
from data_loader import Data
import constants as C
from models import resnet18

# from EDGE_4_4_1 import EDGE
# from matplotlib.ticker import MaxNLocator
# plt.style.use('ggplot')
log = logging.getLogger("sampleLogger")

class Network():
    def __init__(self, device, arch, num_classes=10, pretrained=True,
                 feature_extracting=False):
        """Setup the network and adjust it to the dataset dimentions.
        Parameters
        ----------
        device: str
            "cpu" or "cuda"
        arch: str
            network architecture we want to use, "vgg16", "resnet", "alexnet"
        num_classes: int
            number of classes in the dataset
        pretrained: bool
            want the network to be pretrained or not
        feature_extracting: bool
            whether we want to only do feature extraction or network
        finetuning. This is used for adjusting the layers requires_grad
        attribute.
        """
        self.preprocess = None
        self.model = None
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = device
        self.feature_extracting = feature_extracting
    
    def set_model(self):
        if self.arch == "vgg11":
            if self.pretrained == "True":
                self.model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.vgg11()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "vgg16":
            if self.pretrained == "True":
                self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.vgg16()
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "resnet":
            if self.pretrained == "True":
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.resnet18()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            # self.model = resnet18()

        elif self.arch == "alexnet":
            if self.pretrained == "True":
                self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.alexnet()
            num_ftrs = self.model.fc.in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        else:
            sys.exit("Wrong architecture")

        self.model = self.model.to(self.device)
        return self.model


    def set_parameter_requires_grad(self):
        if self.feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False


    def trained_enough(self, accuracy, dataloader, loss_fn, optimizer, epochs, device):
        i = 0
        while accuracy < .20:
            accuracy, _ = train(self.model, dataloader, loss_fn, optimizer,
                                epochs, device)
            log.debug(f"{i} epoch extra training, accuracy: {100 * accuracy}")
            i += 1

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

def train(model, dataloader, loss_fn, optimizer, epochs, device):
    log.debug('Training...')
    size = len(dataloader.dataset)

    for t in range(epochs):
        log.debug(f"Epoch {t+1}")
        correct = 0
        running_loss = 0.
        last_loss = 0.

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Set frozen param grads to 0.
            # pruner.make_grads_zero()

            optimizer.step()

            running_loss += loss.item()

            if batch % 100 == 0:
                last_loss, current = running_loss / 100, batch * len(X)
                log.debug(f"loss: {last_loss:>3f}  [{current:>5d}/{size:>5d}]")
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        log.debug(f"Training Error: Accuracy: {(100*correct):>0.1f}%")
    return 100.0 * correct, loss


def test(model, dataloader, loss_fn, device):
    log.debug('Testing')
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
    log.debug(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    model.train()
    return 100. * correct



def main():
    args = utils.get_args()
    logger = utils.setup_logger()

    # preparing the hardware
    device = utils.get_device()

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for i in range(3):
        train_acc = network.train(train_dl, loss_fn, optimizer,
                                  args.train_epochs) 
        test_acc = network.test(test_dl, loss_fn)


if __name__ == '__main__':
    main()

