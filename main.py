#system import
import os
import sys
import time
from datetime import datetime
import argparse
from copy import deepcopy
import glob
import random
import json
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from videoloaders.transforms_video import ToTensorVideo,ResizeVideo,RandomCropVideo,CenterCropVideo,NormalizeVideo
from videoloaders.transform_temporal import TemporalTransform
from something_dataset import Something
from metrics.AverageMeter import AverageMeter
from metrics.accuracy import accuracy

def setup_device(gpu_id):
    #set up GPUS
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if  int(gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print("set CUDA_VISIBLE_DEVICES=",gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device %s"%device)
    return device

def save_json(dict,path):
    with open(path, 'w') as f:
        json.dump( dict, f, sort_keys=True, indent=4)
        print("log saved at %s"%path)
        
def save_checkpoint(path,model,key="model"):
    #save model state dict
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("checkpoint saved at",path)

def setup_args():
    parser = argparse.ArgumentParser(description="")    
    parser.add_argument('--somethingroot', type=str, default="./data/something", help = "root path of 20bn-something-something-v2 that has all .webm files")
    parser.add_argument('--train', type=str, default="./data/example.train.csv", help = "train dataset txt")
    parser.add_argument('--val', type=str,default = "./data/example.val.csv", help = "val dataset txt")
    parser.add_argument('--test', type=str,default = "./data/example.test.csv", help = "test dataset txt")
    parser.add_argument('--workers', type=int, default=8, help="number of processes to make batch worker.")
    parser.add_argument('--gpu','-g', type=int, default=-2,help = "gpu id. -1 means cpu.")
    parser.add_argument('--optimizer', type=str, default="adam",choices = ["adam","sgd"], help = "optmizer")
    parser.add_argument('--lr', type=float, default=1e-4, help = "learning rate.")
    parser.add_argument('--decay', type=float, default=1e-3, help = "weight decay.")
    parser.add_argument('--patience', type=int, default=2, help="patience")
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to decrease learning rate when val acc stop improving")
    parser.add_argument('--lr-min', type=float, default=1e-5, help = "if lr becomes less than this, stop")
    parser.add_argument('--batch', type=int, default=15, help="batch size")
    parser.add_argument('--epochs', type=int, default=100, help="maximum number of epochs. if 0, evaluation only")
    return parser.parse_args()

        
def setup_dataset(args):
    #setup dataset as pandas data frame
    df_dict = {}
    df_dict["train"] = pd.read_csv(args.train, sep=',')
    df_dict["val"] = pd.read_csv(args.val, sep=',')
    df_dict["test"] = pd.read_csv(args.test, sep=',')
    
    label_key = "verb_class_name"
    
    #use kinetics mean and std
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    #key is train/val/test and the value is corresponding pytorch dataset
    dataset_dict = {}
    
    #target_transform
    #target_transform is mapping from category name to category idx start from 0
    assert len(df_dict["train"][label_key].unique())==len(df_dict["test"][label_key].unique())
    assert len(df_dict["train"][label_key].unique())==len(df_dict["val"][label_key].unique())
    #assert len(df_dict["train"][label_key].unique())==len(df_dict["test-same"][label_key].unique())
    target_transform =  {v:i for i,v in enumerate(sorted(df_dict["train"][label_key].unique()))}
    
    for split,df in df_dict.items():
        print("setting up",split,"dataset")
        #tempral transform 
        #tempral transform is to make the save lengthed clip
        if split=="train":
            temporal_transform = TemporalTransform(length=64,mode="random") 
        elif split=="val":
            temporal_transform = TemporalTransform(length=64,mode="center") 
        elif split=="test":
            temporal_transform = TemporalTransform(length=64,mode="center")
        else:
            raise NotImplementedError() 
        
        if split=="train":
            transform = transforms.Compose([
                            ToTensorVideo(),
                            ResizeVideo(w=148,h=112),
                            RandomCropVideo(112),
                            NormalizeVideo(mean,std,inplace=True),
                            ])
        else:
            transform = transforms.Compose([
                            ToTensorVideo(),
                            ResizeVideo(w=148,h=112),
                            CenterCropVideo(112),
                            NormalizeVideo(mean,std,inplace=True),
                            ])
        dataset_dict[split] = Something(df,
                                        label_key=label_key,
                                        root=args.somethingroot,
                                         temporal_transform = temporal_transform,
                                         transform=transform,
                                         target_transform=target_transform,
                                        )
            
    return dataset_dict

def setup_dataloader(args,dataset_dic):
    dataloader_dict = {}
    for split,dataset in dataset_dic.items():
        dataloader_dict[split] = DataLoader(dataset,
                                          batch_size=args.batch, 
                                          shuffle= split=="train",
                                          num_workers=args.workers,
                                          pin_memory=True,
                                         )
    return dataloader_dict

def setup_backbone(num_classes=5):
    r3d = torchvision.models.video.r3d_18(pretrained=True)
    r3d.fc = nn.Linear(512,num_classes)
    r3d = nn.Sequential(OrderedDict([
            ('feature',nn.Sequential(*list(r3d.children())[:-1]+[nn.Flatten()])),
            ('classifier',list(r3d.children())[-1]),
        ]))
    return r3d 
    
def train_one_epoch(dataloader,model,criterion,optimizer,accuracy=accuracy,device=None,print_freq=100):
    since = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # Set model to training mode
    
    losses = AverageMeter()
    accs = AverageMeter()
   
    for i,data in enumerate(tqdm(dataloader)):
        inputs = data["input"].to(device)
        labels = data["label"].to(device)

        feat = model.feature(inputs)
        outputs = model.classifier(feat)        
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), outputs.size(0))
        accs.update(acc.item(),outputs.size(0))

        if i % print_freq == 0 or i == len(dataloader)-1:
            temp = "current loss: %0.5f "%loss.item()
            temp += "acc %0.5f "%acc.item()
            temp += "| running average loss %0.5f "%losses.avg
            temp += "acc %0.5f "%accs.avg
            print(i,temp)

    time_elapsed = time.time() - since
    print('this epoch took {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return float(losses.avg),float(accs.avg)

def evaluate(dataloader,model,criterion,accuracy,device=None,keep_output=False,use_softmax=True):
    softmax = torch.nn.Softmax(dim=1)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_outputs = []
    all_labels = []
    all_ids = []
    
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        for i,data_ in enumerate(tqdm(dataloader)):
            data = deepcopy(data_)#see https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            inputs = data["input"].to(device)
            labels = data["label"]
            ids = data['id']
            
            if keep_output:
                all_labels.append(labels)
                all_ids+=ids.tolist()
            labels = labels.to(device)
            
            feat = model.feature(inputs)
            outputs = model.classifier(feat)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            losses.update(loss.item(), outputs.size(0))
            accs.update(acc.item(),outputs.size(0))
            
            
            if keep_output:
                if use_softmax:
                    outputs = softmax(outputs)
                all_outputs.append(outputs.cpu())
                
        if keep_output:
            all_outputs = torch.cat(all_outputs).tolist()
            all_labels = torch.cat(all_labels).tolist()
                        
    print("eval loss %0.5f acc %0.5f "%(losses.avg,accs.avg))    
    if keep_output:
        return float(losses.avg),float(accs.avg),all_outputs,all_labels,all_ids
    else:
        return float(losses.avg),float(accs.avg)

def group_decay_nodecay_params(model):
    params_nodecay = []
    params_decay = []
    for pname, param in model.named_parameters():
        module_tree = pname.split(".")[0:-1]
        ptype = pname.split(".")[-1]

        #e.g. go to layer4.2.conv3 from layer4.2.conv3.weight
        module = model
        for mname in module_tree:
            module = module._modules[mname]

        # print(module,pname)
        if (isinstance(module, nn.Linear) or isinstance(module, nn.modules.conv._ConvNd)):
            if ptype=="bias":
                #print(module,pname,"is not decayed")
                params_nodecay.append(param)
            else:
                params_decay.append(param)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm): 
            #print(module,pname,"is not decayed")
            params_nodecay.append(param)
        else:
            params_decay.append(param)

    assert len(params_decay)+len(params_nodecay) == sum(1 for k in model.parameters())
    groups = [dict(params=params_decay), dict(params=params_nodecay, weight_decay=.0)]
    return groups

def remove_no_grad_params(params):
    '''
    params: list of dict, which can be output of group_decay_nodecay_params(model)
    '''
    for param in params:
        param["params"] = [p for p in param["params"] if p.requires_grad]
    return params

def setup_optimizer(model,algorithm,lr,weight_decay,filter_bias=True):
    if filter_bias:
        params = group_decay_nodecay_params(model)
    else:
        params = [{"params":model.parameters()}]
    params = remove_no_grad_params(params)
        
    if algorithm == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9,weight_decay=weight_decay)
    elif algorithm == "adam":
        optimizer = optim.Adam(params, lr=lr,amsgrad=True,weight_decay=weight_decay)
    else:
        raise NotImplementedError("not implemented %s option"%algorithm)
    return optimizer
    
def main(args):
    since = time.time()
    print(args)
    
    #setup logdir
    log_dir = "./logs"

    #setup cuda device
    device = setup_device(args.gpu)
        
    #setup dataset and dataloaders
    dataset_dict = setup_dataset(args)
    dataloader_dict = setup_dataloader(args,dataset_dict)
            
    #setup backbone cnn
    num_classes = dataset_dict["train"].num_classes
    model = setup_backbone(num_classes=num_classes)

    #setup loss and acc
    criterion = torch.nn.CrossEntropyLoss().to(device)

    #setup optimizer
    optimizer = setup_optimizer(model,algorithm=args.optimizer,lr=args.lr,weight_decay=args.decay,filter_bias=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.patience, factor=args.step_facter,verbose=True)
    
    #main training
    log = {}
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log["train"] = []
    log["val"] = []
    log["lr"] = []
    log_save_path = os.path.join(log_dir,"log.json")
    save_json(log,log_save_path)
    valacc = 0
    best_val_acc = 0
    bestmodel = model
    for epoch in range(args.epochs):
        print("epoch: %d --start from 0 and at most end at %d"%(epoch,args.epochs-1))            
        loss,acc = train_one_epoch(dataloader_dict["train"],model,criterion,
                        optimizer,accuracy=accuracy,
                        device=device,print_freq=100)
        
        log["train"].append({'epoch':epoch,"loss":loss,"acc":acc})
        
        valloss,valacc = evaluate(dataloader_dict["val"],model,criterion,accuracy=accuracy,device=device)
        log["val"].append({'epoch':epoch,"loss":valloss,"acc":valacc})
        lr_scheduler.step(1 - valacc)

        #if this is the best model so far, keep it on cpu and save it
        if valacc > best_val_acc:
            best_val_acc = valacc
            log["best_epoch"] = epoch
            log["best_acc"] = best_val_acc
            bestmodel = deepcopy(model)
            bestmodel.cpu()
            save_path = os.path.join(log_dir,"bestmodel.pth")
            save_checkpoint(save_path,bestmodel,key="model")

        save_json(log,log_save_path)
        max_lr_now = max([ group['lr'] for group in optimizer.param_groups ])
        log["lr"].append(max_lr_now)
        if max_lr_now < args.lr_min:
            break
            
    #use the best model to evaluate on test set
    print("test started")
    loss,acc,all_outputs,all_labels,all_ids  = evaluate(dataloader_dict["test"],bestmodel,criterion,accuracy=accuracy,
                                                device=device,keep_output=True,use_softmax=True)
    log["test"] = {"loss":loss,"acc":acc,"all_labels":all_labels,"all_outputs":all_outputs,"all_ids":all_ids}
    
    time_elapsed = time.time() - since
    log["time_elapsed"] = time_elapsed
    #save the final log
    save_json(log,log_save_path)
        
if __name__ == '__main__':
    args = setup_args()
    main(args)