import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import copy

def get_splitCIFAR(seed=0, pc_valid=0.10, task_num = 0):

    if os.path.isfile(("../data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)


    data={}
    data = dict.fromkeys(['train','valid','test'])
    for s in ['train','valid','test']:
        data[s]={'x':[],'y':[]}
        data[s]['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('x_'+s+'.bin')))
        data[s]['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('y_'+s+'.bin')))

    return data
    
    
    
def make_splitcifar(seed=0, pc_valid=0.2):
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR10('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='cifar10'
    data[0]['ncla']=10
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(target.numpy()[0])
    
    # "Unify" and save
    for s in ['train','test']:
        data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
        data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    
    # CIFAR100
    dat={}
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    for n in range(1,11):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=10
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            task_idx = target.numpy()[0] // 10 + 1
            data[task_idx][s]['x'].append(image)
            data[task_idx][s]['y'].append(target.numpy()[0]%10)
    
    
    
    for t in range(1,11):
        for s in ['train','test']:
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
            
    os.makedirs('../data/split_cifar/' ,exist_ok=True)
    
    for t in range(0,11):
      # Validation
      r=np.arange(data[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/split_cifar/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/split_cifar/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/split_cifar/'+ str(t) + '/y_' + s + '.bin'))
    
    
    
    
    
    
    
    
    
    
    
    

def get_CIFARsubset(seed=0, pc_valid=0.20, task_num = 0):

    if os.path.isfile(("../data/sim_cifar100/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_subsets(seed=seed, pc_valid=pc_valid)

    data={}
    data = dict.fromkeys(['train','valid','test'])
    for s in ['train','valid','test']:
        data[s]={'x':[],'y':[]}
        data[s]['x']=torch.load(os.path.join(os.path.expanduser(('../data/sim_cifar100/' + str(task_num))), ('x_'+s+'.bin')))
        data[s]['y']=torch.load(os.path.join(os.path.expanduser(('../data/sim_cifar100/' + str(task_num))), ('y_'+s+'.bin')))

    return data    
    
    
    
def make_subsets(seed=0, pc_valid=0.1):
        
    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_labels = [[],[],[],[],[],[]]
    test_labels = [[],[],[],[],[],[]]
    ### Keys indicating which fine labels from CIFAR100 will be included in each of the data subsets
    keys = [["bicycle", "bus", "motorcycle", "pickup_truck","train","lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            ["beaver","dolphin","otter","seal","whale","aquarium_fish","flatfish","ray", "shark","trout"],
            ["bed","chair","couch","table","wardrobe","clock","keyboard","lamp","telephone","television"],
            ["hamster","mouse","rabbit","shrew","squirrel","fox","porcupine","possum","raccoon","skunk"],
            ["orchid","poppy","rose","sunflower","tulip","apple","mushroom","orange","pear","sweet_pepper"],
            ["crab","lobster","snail","spider","worm","bee","beetle","butterfly","caterpillar","cockroach"]]
    
    os.makedirs('../data/split_cifar100', exist_ok =True)
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
      data[t]={}
      data[t]['name']='cifar100'
      data[t]['ncla']=10
      data[t]['train']={'x': [],'y': []}
      data[t]['test']={'x': [],'y': []}
      train_labels[t] = [dat['train'].class_to_idx[k] for k in keys[t]]
      test_labels[t] = [dat['test'].class_to_idx[k] for k in keys[t]]
    
    ### Extract only the appropriately labeled samples for each of the subsets
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:      
          for t in range(0,6):
            if target in (train_labels[t]):
              data[t][s]['x'].append(image)
              data[t][s]['y'].append(target.numpy()[0])
          
    # Deep copy the nested list
    dataprocessed = copy.deepcopy(data)
    
    ### Convert the labels of each subset to range from 0-9
    for t in range(0,6):
      for l in range(0,10):
        for i in range(len(data[t]["train"]["y"])):
          if data[t]["train"]["y"][i] == train_labels[t][l]:
            dataprocessed[t]["train"]["y"][i] = l
    
        for i in range(len(data[t]["test"]["y"])):
          if data[t]["test"]["y"][i] == test_labels[t][l]:
            dataprocessed[t]["test"]["y"][i] = l
    
    os.makedirs("../data/sim_cifar100",exist_ok=True)
    
    for t in range(0,6):
        for s in ['train','test']:
            dataprocessed[t][s]['x']=torch.stack(dataprocessed[t][s]['x']).view(-1,size[0],size[1],size[2])
            dataprocessed[t][s]['y']=torch.LongTensor(np.array(dataprocessed[t][s]['y'],dtype=int)).view(-1)

    
    ### Splitting validation off from training rather than test is fine here, so long as both sets are preprocessed identically
    for t in range(0,6):
      # Validation
      r=np.arange(dataprocessed[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      dataprocessed[t]['valid']={}
      dataprocessed[t]['valid']['x']=dataprocessed[t]['train']['x'][ivalid].clone()
      dataprocessed[t]['valid']['y']=dataprocessed[t]['train']['y'][ivalid].clone()
      dataprocessed[t]['train']['x']=dataprocessed[t]['train']['x'][itrain].clone()
      dataprocessed[t]['train']['y']=dataprocessed[t]['train']['y'][itrain].clone()
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/sim_cifar100/' + str(t)) ,exist_ok=True)
        torch.save(dataprocessed[t][s]['x'], ('../data/sim_cifar100/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(dataprocessed[t][s]['y'], ('../data/sim_cifar100/'+ str(t) + '/y_' + s + '.bin'))
    