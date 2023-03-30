import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0, pc_valid=0.20, task_num = 0):
    data={}
    taskcla=[]
    size=[3,32,32]

    # if not os.path.isdir('../data/binary_cifar10/'):
    #     # CIFAR10
    #     os.makedirs('../data/binary_cifar10')

    #     mean=[x/255 for x in [125.3,123.0,113.9]]
    #     std=[x/255 for x in [63.0,62.1,66.7]]
        
    #     # CIFAR10
    #     dat={}
    #     dat['train']=datasets.CIFAR10('../data/',train=True,download=True,
    #                                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    #     dat['test']=datasets.CIFAR10('../data/',train=False,download=True,
    #                                  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    #     data[0]={}
    #     data[0]['name']='cifar10'
    #     data[0]['ncla']=10
    #     data[0]['train']={'x': [],'y': []}
    #     data[0]['test']={'x': [],'y': []}
    #     for s in ['train','test']:
    #         loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    #         for image,target in loader:
    #             data[0][s]['x'].append(image)
    #             data[0][s]['y'].append(target.numpy()[0])
        
    #     # "Unify" and save
    #     for s in ['train','test']:
    #         data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
    #         data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    #         torch.save(data[0][s]['x'], os.path.join(os.path.expanduser('../data/binary_cifar10'),'data'+s+'x.bin'))
    #         torch.save(data[0][s]['y'], os.path.join(os.path.expanduser('../data/binary_cifar10'),'data'+s+'y.bin'))
    
    # if not os.path.isdir('../data/binary_split_cifar100/'):
    #     # CIFAR100
    #     os.makedirs('../data/binary_split_cifar100')
    #     dat={}
        
    #     mean = [0.5071, 0.4867, 0.4408]
    #     std = [0.2675, 0.2565, 0.2761]
        
    #     dat['train']=datasets.CIFAR100('../data/',train=True,download=True,
    #                                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    #     dat['test']=datasets.CIFAR100('../data/',train=False,download=True,
    #                                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    #     for n in range(1,11):
    #         data[n]={}
    #         data[n]['name']='cifar100'
    #         data[n]['ncla']=10
    #         data[n]['train']={'x': [],'y': []}
    #         data[n]['test']={'x': [],'y': []}
    #     for s in ['train','test']:
    #         loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    #         for image,target in loader:
    #             task_idx = target.numpy()[0] // 10 + 1
    #             data[task_idx][s]['x'].append(image)
    #             data[task_idx][s]['y'].append(target.numpy()[0]%10)

        
        
    #     for t in range(1,11):
    #         for s in ['train','test']:
    #             data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
    #             data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
    #             torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
    #                                                      'data'+str(t)+s+'x.bin'))
    #             torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
    #                                                      'data'+str(t)+s+'y.bin'))
    
    
    
    
    # Load binary files
    data={}
    
    data = dict.fromkeys(['train','test'])
    for s in ['train','test']:
        data[s]={'x':[],'y':[]}
        if task_num == 0:
            data[s]['x']=torch.load(os.path.join(os.path.expanduser('../data/binary_cifar10'),'data'+s+'x.bin'))
            data[s]['y']=torch.load(os.path.join(os.path.expanduser('../data/binary_cifar10'),'data'+s+'y.bin'))
        else:
            data[s]['x']=torch.load(os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                    'data'+str(task_num)+s+'x.bin'))
            data[s]['y']=torch.load(os.path.join(os.path.expanduser('../data/binary_split_cifar100'),
                                                    'data'+str(task_num)+s+'y.bin'))

            
    # Validation
    r=np.arange(data['train']['x'].size(0))
    r=np.array(shuffle(r,random_state=seed),dtype=int)
    nvalid=int(pc_valid*len(r))
    ivalid=torch.LongTensor(r[:nvalid])
    itrain=torch.LongTensor(r[nvalid:])
    data['valid']={}
    data['valid']['x']=data['train']['x'][ivalid].clone()
    data['valid']['y']=data['train']['y'][ivalid].clone()
    data['train']['x']=data['train']['x'][itrain].clone()
    data['train']['y']=data['train']['y'][itrain].clone()

    return data