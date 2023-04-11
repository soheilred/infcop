import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.init as init
import pickle

import utils
from data_loader import Data
from network import Network, train, test
from correlation import Activations
import constants as C
import logging
import logging.config

log = logging.getLogger("sampleLogger")

class Pruner:
    def __init__(self, model, prune_percent,
                 train_dataloader=None, test_dataloader=None,
                 composite_mask=None, all_task_mask=None):
        """Prune the network.
        Parameters
        ----------
        model: Network
            The network to be pruned
        prune_percent: int
            The percent to which each layer of the network is pruned
        train_dataloader: dataloader
        test_dataloader: dataloader
        composite_mask:
            The composite mask stores the task number for which every weight was
            frozen, or if they are unfrozen the number is the current task 
        all_task_masks: dict
            A dictionary of binary masks, 1 per task, indicating
            which weights to include when evaluating that task 
        """
        self.model = model
        self.mask = None
        self.prune_percent = prune_percent
        self.reinit = False
        self.num_layers = 0
        self.task_num = 0
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.init_state_dict = None
        self.composite_mask = composite_mask
        self.all_task_masks = all_task_masks 

        # Initialize Weights 
        self.model.apply(self.weight_init)
        # Making Initial Mask
        self.init_mask()

    def weight_init(self, m):
        '''Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    def initialize_lth(self):
        """Prepare the lth object by:
        1. initializing the network's weights
        2. saving the initial state of the network into the object
        3. saving the initial state model on the disk
        4. initializing the masks according to the layers size
        """

        # Weight Initialization
        self.model.apply(self.weight_init)

        # Copying and Saving Initial State
        self.init_state_dict = copy.deepcopy(self.model.state_dict())
        utils.save_model(self.model, C.MODEL_ROOT_DIR, "/initial_model.pth.tar")

        # Making Initial Mask
        self.init_mask()
        return self.init_state_dict

    def count_layers(self):
        count = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                count = count + 1
        return count

    def init_mask(self):
        """Make an empty mask of the same size as the model."""
        self.num_layers = self.count_layers()
        self.mask = [None] * self.num_layers
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[step] = np.ones_like(tensor)
                step = step + 1

    def reset_weights_to_init(self, initial_state_dict):
        """Reset the remaining weights in the network to the initial values.
        """
        step = 0
        mask_temp = self.mask
        for name, param in self.model.named_parameters():
            if "weight" in name:
                weight_dev = param.device
                param.data = torch.from_numpy(mask_temp[step] *
                                              initial_state_dict[name].
                                              cpu().numpy()).to(weight_dev)
                step = step + 1
            if "bias" in name:
                param.data = initial_state_dict[name]

    def prune_by_percentile(self):
        # Calculate percentile value
        step = 0
        for name, param in self.model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), self.prune_percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0,
                                    self.mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[step] = new_mask
                step += 1

    def prune_once(self, initial_state_dict):
        step = 0
        self.prune_by_percentile()
        # if the reinit option is activated
        if self.reinit:
            self.model.apply(self.weight_init)
            step = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy() *
                                                  self.mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
        else:
            # todo: why do we need an initialization here?
            self.reset_weights_to_init(initial_state_dict)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0.
            During training this is called to avoid storing gradients for the
            frozen weights, to prevent updating.
            This is unaffected in the shared masks since shared weights always
            have the current index unless frozen 
        """
        # assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.task_num)] = 0
                if self.task_num>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)
                    
            elif 'BatchNorm' in str(type(module)) and self.task_num>0:
                    # Set grads of batchnorm params to 0.
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Set all pruned weights to 0.
            This is just a prune() but with pre-calculated masks
        """
        # assert self.current_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.composite_mask[module_idx]
                module.weight.data[layer_mask.gt(self.task_num)] = 0.0

    def apply_mask(self):
        """Applies appropriate mask to recreate task model for inference.
            To be done to retrieve weights just for a particular dataset
        """
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = -100

                # Any weights which weren't frozen in one of the tasks before
                # or including task # dataset_idx are set to 0 
                for i in range(0, self.task_num + 1):
                    if i == 0:
                        mask = self.all_task_masks[i][0][module_idx].cuda()
                    else:
                        mask = mask.logical_or(self.all_task_masks[i][0][module_idx].cuda())
                weight[mask.eq(0)] = 0.0

        self.model.eval()
   
    def increment_task(self):
        """
            Turns previously pruned weights into trainable weights for
            current dataset.
            Also updates task number and prepares new task mask
        """
        self.task_num += 1
                
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                task = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    task = task.cuda()
                task_mask[module_idx] = task

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[self.task_num] = [task_mask]

        print("Exiting finetuning mask")

    def prune(self):
        """Apply the masks to the weights.
            Goes through and calls prune_mask for each layer and stores the results
            and then applies the masks to the weights
        """
        print('Pruning for dataset idx: %d' % (self.task_num))
        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
    
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### Get the pruned mask for the current layer
                mask = self.pruning_mask(module.weight.data,
                                         self.composite_mask[module_idx],
                                         module_idx)
                self.composite_mask[module_idx] = mask.cuda()

                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.composite_mask[module_idx].gt(self.task_num)] = 0.0
                self.all_task_masks[self.task_num][0][module_idx][mask.gt(self.task_num)] = 0

                
        print("\nFOR TASK %d:", self.task_num)
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_frozen = weight[self.composite_mask[layer_idx].eq(self.task_num)].numel()
                print('Layer #%d: Frozen %d/%d (%.2f%%)' %
                      (layer_idx, num_frozen, num_params,
                       100 * num_frozen / num_params))                

 
    def pruning_mask(self, weights, composite_mask_in, layer_idx):
        """Rank prunable filters by magnitude.
            Sets all below kth to 0.
            Returns pruned mask.
        """
        composite_mask = composite_mask_in.clone().detach().cuda()
        filter_weights = weights
        import ipdb; ipdb.set_trace()
        filter_composite_mask = composite_mask.eq(self.task_num)
        tensor = weights[filter_composite_mask]

        abs_tensor = tensor.abs()


        # Code for increasing the freezing percent of a given layer based on
        # connectivity 

        prune_rank = round(self.prune_perc * abs_tensor.numel())
        connlist = self.conn_aves[self.task_num]


        max_n_layers_indices = np.argsort(list(connlist.values()))
        max_n_keys = np.asarray(list(connlist.keys()))[list(max_n_layers_indices)]
        random_idxs = np.copy(max_n_keys)
        np.random.shuffle(random_idxs)
        
        # ### Apply freezing if the index is selected based on connectivity, otherwise prune at the baseline rate.
        # if self.freeze_order == "top" and (layer_idx in max_n_keys[-self.num_freeze_layers:]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        # elif self.freeze_order == "bottom" and (layer_idx in max_n_keys[:self.num_freeze_layers]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        # elif self.freeze_order == "random" and (layer_idx in random_idxs[:self.num_freeze_layers]):
        #     prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        
        prune_value = abs_tensor.view(-1).cpu().kthvalue(prune_rank)[0]

        remove_mask = torch.zeros(weights.shape)

        remove_mask[filter_weights.abs().le(prune_value)]=1
        remove_mask[composite_mask.ne(self.task_num)]=0

        composite_mask[remove_mask.eq(1)] = self.task_num + 1
        mask = composite_mask
        
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.gt(self.task_num).sum(), tensor.numel(),
              100 * mask.gt(self.task_num).sum() / tensor.numel(), weights.numel()))

        return mask
     
    def train(self, dataloader, loss_fn, optimizer, epochs, device):
        log.debug('Training...')
        size = len(dataloader.dataset)

        # Get optimizer with correct params.
        params_to_optimize = self.model.parameters()
        optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9,
                              weight_decay=self.args.weight_decay,
                              nesterov=True) 
        scheduler = MultiStepLR(optimizer, milestones=self.args.Milestones,
                                gamma=self.args.Gamma)     

        ### Note: After the first task, batchnorm and bias throughout the
        ### network are frozen, this is what train_nobn() refers to 
        if self.task_num > 0:
            self.model.train_nobn()
            print("No BN in training loop")

        for t in range(epochs):
            log.debug(f"Epoch {t+1}")
            correct = 0
            running_loss = 0.
            last_loss = 0.

            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Set frozen param grads to 0.
                self.make_grads_zero()

                optimizer.step()

                # Set pruned weights to 0.
                self.pruner.make_pruned_zero()

                running_loss += loss.item()

            scheduler.step()


                if batch % 100 == 0:
                    last_loss, current = running_loss / 100, batch * len(X)
                    log.debug(f"loss: {last_loss:>5f}  [{current:>5d}/{size:>5d}]")
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= size
            log.debug(f"Training Error: Accuracy: {(100*correct):>0.1f}%")
        return 100.0 * correct, loss


    def find_unstable_layers(self, control_corrs):
        control_corrs = np.array(control_corrs[1])
        # mid = abs(np.median(control_corrs[1]))
        thrsh = 100
        unstable_layers = np.where(np.abs(control_corrs) > thrsh)[0]
        return unstable_layers


    def controller(self, control_corrs, layers_dim, cont_type, imp_iter, cont_layer_list=None):
        # calculate the control proportional coefficient
        log.debug(f"apply controller at layer {cont_layer_list}")

        # coef = control_corrs[0][cont_layer_list] / max(control_corrs[1][layer_num], eps)
        if cont_layer_list is None:
            cont_layer_list = self.find_unstable_layers(control_corrs)

        #     log.debug(f"controller coef at layer {ind}: {coef}")
        #     contr_mask[ind] = (control_weights[ind] * coef).astype("float32")

        # get the weights from previous iteration
        prev_iter_weights = self.get_prev_iter_weights(imp_iter, cont_layer_list)

        # get connectivity
        connectivity = [(torch.mean(control_corrs[imp_iter - 1][i]).item() /
                        (layers_dim[i][0] * layers_dim[i + 1][0]))
                        for i in range(len(layers_dim) - 1)]

        # get the coefficient based on connectivity
        for ind in cont_layer_list:
            prev_corr = self.get_prev_iter_correlation(control_corrs, layers_dim,
                                                         imp_iter, ind)
            prev_weight = prev_iter_weights[ind]

            # type 1
            if (cont_type == 1):
                control_weights = prev_corr

            # type 2
            elif (cont_type == 2):
                control_weights = torch.mul(prev_corr, prev_weight)

            # type 3
            elif (cont_type == 3):
                control_weights = connectivity[ind] * prev_weight

            self.apply_controller(control_weights, ind)

        # self.apply_controller(control_weights=control_corrs, layer_list=layer_list)


    def get_prev_iter_correlation(self, control_corrs, layers_dim, imp_iter, ind):
        # the + 1 is for matching to the connectivity's dimension
        weights = control_corrs[imp_iter - 1][ind - 1]
        kernel_size = layers_dim[ind][-1]
        weights = weights.tile(dims=(kernel_size, kernel_size, 1, 1)).\
                               transpose(1, 2).transpose(0, 3)
                               # transpose(1, 2).transpose(0, 3).transpose(0, 1)
        return weights


    def get_prev_iter_weights(self, imp_iter, layers_list):
        model = torch.load(C.MODEL_ROOT_DIR + str(imp_iter) + '_model.pth.tar')
        # model.to(device)
        model.eval()
        weights = {}

        ind = 0
        for name, param in model.named_parameters():
            if ("weight" in name and 
               ("conv" in name or "fc" in name or "features" in name)):
                if ind in layers_list:
                    log.debug(f"weights at layer {ind} in iteration {imp_iter} is added")
                    weights[ind] = param.data
                ind += 1
            if ind > max(layers_list):
                break

        return weights


    def apply_controller(self, control_weights, layer_ind):
        ind = 0
        # get a handle to the layer's weights
        for name, param in self.model.named_parameters():
            # if "weight" in name:
            if ("weight" in name and 
               ("conv" in name or "fc" in name or "features" in name)):
                if ind == layer_ind:
                    # weight = param.data.cpu().numpy()
                    weight = param.data
                    # exp = np.exp(weight)
                    weight_dev = param.device
                    # contr_mask = (np.ones(weight.shape) * coef).astype("float32")
                    # param.data = torch.from_numpy(weight * exp * contr_mask).to(weight_dev)
                    # new_weights = utils.batch_mul(weight, control_weights)
                    new_weights = torch.mul(weight, control_weights)
                    param.data = new_weights.to(weight_dev)
                    # param.data = torch.from_numpy(weight * control_weights).to(weight_dev)
                    break
                ind += 1
        # weight = self.model.features[layer_list].weight.data.cpu().numpy()
        # cur_weights = layer_param.data.cpu().numpy()



def def main():
    ():
    # preparing the hardware
    device = utils.get_device()
    args = utils.get_args()
    logger = utils.setup_logger()
    num_exper = 5

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = args.imp_iter               # 35 was the default
    MODEL_DIR = MODEL_ROOT_DIR + arch_type + "/" + args.dataset + "/"

    # Copying and Saving Initial State
    corr = []

    pruning = LTPruning(model, args.arch, args.prune_perc_per_layer, train_dl,
                        test_dl)

if __name__ == '__main__':
    main()
