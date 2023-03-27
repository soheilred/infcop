import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg11, vgg16,\
    alexnet, VGG11_Weights, VGG16_Weights, AlexNet_Weights

import logging
# from EDGE_4_4_1 import EDGE
# from matplotlib.ticker import MaxNLocator
# plt.style.use('ggplot')
log = logging.getLogger("sampleLogger")

class VGG16(nn.Module):
    def __init__(self, num_channels=3, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x, control):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.control_function(out, control)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def control_function(self, x, control):
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.n_hidden = [128, 64, 32, 16, 10]
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(28 * 28, self.n_hidden[0])
        self.fc1 = nn.Linear(self.n_hidden[0], self.n_hidden[4])
        # self.fc2 = nn.Linear(self.n_hidden[1], self.n_hidden[2])
        # self.fc3 = nn.Linear(self.n_hidden[2], self.n_hidden[3])
        # self.fc4 = nn.Linear(self.n_hidden[3], self.n_hidden[4])

    def forward(self, x):
        x = self.flatten(x) # flatten all dimensions except the batch
        x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc1(x)
        return x


class Network():
    def __init__(self, device, arch, pretrained=False):
        "docstring"
        self.preprocess = None
        self.model = None
        self.arch = arch
        self.pretrained = pretrained
        self.device = device

    def set_model(self):
        if self.arch == "vgg11":
            if self.pretrained:
                weights = VGG11_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = vgg11(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = vgg11().to(self.device)

        elif self.arch == "vgg16":
            if self.pretrained:
                weights = VGG16_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = vgg16(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = vgg16().to(self.device)

        elif self.arch == "resnet":
            if self.pretrained:
                weights = ResNet50_Weights.DEFAULT
                self.preprocess = weights.transforms()
                self.model = resnet50(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = resnet50().to(self.device)

        elif self.arch == "alexnet":
            if self.pretrained:
                weights = AlexNet_Weights.IMAGENET1K_V1
                self.preprocess = weights.transforms()
                self.model = alexnet(weights=weights).to(self.device)
                self.model.eval()
            else:
                self.model = alexnet().to(self.device)
        else:
            sys.exit("Wrong architecture")

        return self.model

    # weights = ResNet50_Weights.DEFAULT
    # preprocess = weights.transforms()
    # weights = VGG16_Weights.IMAGENET1K_V1
    # preprocess = VGG16_Weights.IMAGENET1K_V1.transforms
    # model = VGG11(in_channels=3).to(device)
    # model = VGG16(num_channels=3).to(device)
    # model = ResNet(3, block=ResidualBlock, layers=[3, 4, 6, 3]).to(device)
    # model = vgg11().to(device)
    # model = vgg16().to(device)
    # model = resnet50().to(device)
    # model = alexnet().to(device)
    # model = vgg16(weights=weights).to(device)
    # model = resnet50(weights=weights).to(device)
    # model.eval()


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



if __name__ == '__main__':
    main()

