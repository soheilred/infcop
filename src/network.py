import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# from torchvision.models import resnet50, ResNet50_Weights, vgg11, vgg16,\
#     alexnet, VGG11_Weights, VGG16_Weights, AlexNet_Weights

import logging
# from EDGE_4_4_1 import EDGE
# from matplotlib.ticker import MaxNLocator
# plt.style.use('ggplot')
log = logging.getLogger("sampleLogger")

class Network():
    def __init__(self, device, arch, pretrained=False, feature_extracting=False):
        """Setup the network and adjust it to the dataset dimentions.
        Parameters
        ----------
        device: str
            "cpu" or "cuda"
        arch: str
            Network architecture we want to use, "vgg16", "resnet", "alexnet"
        pretrained: bool
            want the network to be pretrained or not
        feature_extracting: bool()
            Whether we want to only do feature extraction or network
        finetuning. This is used for adjusting the layers requires_grad
        attribute.
        """
        self.preprocess = None
        self.model = None
        self.arch = arch
        self.pretrained = pretrained
        self.device = device
    
    def set_model(self):
        if self.arch == "vgg11":
            self.model = models.vgg11(pretrained=self.pretrained)
            self.set_parameter_requires_grad(model, feature_extracting)
            num_ftrs = model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif self.arch == "vgg16":
            self.model = models.vgg16(pretrained=self.pretrained)
            self.set_parameter_requires_grad(model, feature_extracting)
            num_ftrs = model.fc.in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif self.arch == "resnet":
            self.model = models.resnet18(pretrained=self.pretrained)
            self.set_parameter_requires_grad(model, feature_extracting)
            num_ftrs = model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif self.arch == "alexnet":
            self.model = models.alexnet(pretrained=self.pretrained)
            self.set_parameter_requires_grad(model, feature_extracting)
            num_ftrs = model.fc.in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        else:
            sys.exit("Wrong architecture")

        self.model.eval()
        return self.model


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def trained_enough(self, accuracy, dataloader, loss_fn, optimizer, epochs, device):
        i = 0
        while accuracy < .20:
            accuracy, _ = train(self.model, dataloader, loss_fn, optimizer,
                                epochs, device)
            log.debug(f"{i} epoch extra training, accuracy: {100 * accuracy}")
            i += 1

def train(model, dataloader, loss_fn, optimizer, epochs, device):
    log.debug('Training...')
    size = len(dataloader.dataset)
    for t in range(epochs):
        log.debug(f"Epoch {t+1}")
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # log.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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
    return 100. * correct



def main():
    # preparing the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using {device} device")
    if torch.cuda.is_available():
        log.info("Name of the Cuda Device: ", torch.cuda.get_device_name())

    # setting hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    MODEL_DIR = "../data/model/"

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # preparing the training and test dataset
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(training_data, **train_kwargs)
    test_dataloader = DataLoader(test_data, **test_kwargs)

    num_exper = 3
    train_acc = torch.zeros(num_exper)

    for i in range(num_exper):
        torch.manual_seed(i)
        model = NeuralNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_acc[i] = train(model, train_dataloader, loss_fn, optimizer, epochs, device, i)

    # if os.path.exists(MODEL_DIR + "model.pt"):
    #     model = torch.jit.load(MODEL_DIR + "model.pt")
    #     model.eval()


        # test(model, test_dataloader, loss_fn, device)
    # torch.save(model.state_dict(), MODEL_DIR)
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(MODEL_DIR + 'model.pt') # Save
    # layer = [model.linear_stack[2 * i].weight.cpu().detach().numpy() for i in range(5)]
    # for i in range(5 - 1):
    #     print("Computing I(X, Y) between layers", i, "and", i+1)
    #     I[i] = EDGE(layer[i].flatten(), layer[i+1].flatten())
    # print(I)


            # weights = VGG16_Weights.IMAGENET1K_V1
            # self.preprocess = weights.transforms()
            # self.model = vgg16(weights=weights).to(self.device)

                # weights = VGG11_Weights.IMAGENET1K_V1
                # self.preprocess = weights.transforms()
                # self.model = vgg11(weights=weights).to(self.device)

                # weights = ResNet50_Weights.DEFAULT
                # self.preprocess = weights.transforms()
                # self.model = resnet50(weights=weights).to(self.device)

                # weights = AlexNet_Weights.IMAGENET1K_V1
                # self.preprocess = weights.transforms()
                # self.model = alexnet(weights=weights).to(self.device)

if __name__ == '__main__':
    main()

