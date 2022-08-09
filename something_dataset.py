import os
import sys
import glob
import collections
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import skvideo.io  

class Something(Dataset):
    def __init__(self, df,label_key,
                 root = "../data/something/videos/",
                 temporal_transform = None,
                 transform=None,
                 target_transform=None,
                ):
        '''
        df: pandas dataframe of this split
        label_key: column name of storing labels in the daraframe
        root: root path
        temporal_transform: preprpcessing for list of image paths
        transform: preprpcessing for (T, H, W, C) tensor
        target_transform: preprocessing for labels
        '''
        self.df = df
        self.label_key = label_key
        self.root = root
        
        self.temporal_transform = temporal_transform
        self.transform = transform

        self.target_transform = target_transform
        self.num_classes = len(set([self.get_label_idx(i) for i in range(len(self))]))

    def __getitem__(self, i):
        clip = self.get_clip(i)
        outputs = {"input":self.get_clip(i), "label":self.get_label_idx(i),"id":self.df.iloc[i]["id"]}
        return outputs
      
    def get_clip(self,i):
        '''
        get clip
        '''
        info = self.df.iloc[i]
        videodata = skvideo.io.vread(os.path.join(self.root,'%s.webm'%info["id"]))#T,H,W,C
        if self.temporal_transform is not None:
            videodata = self.temporal_transform(videodata)
        videodata = torch.from_numpy(np.stack(videodata))
        if self.transform  is not None:
            videodata = self.transform(videodata)#from (T, H, W, C) to (C, T, H, W)
        return videodata
            
    def get_label(self,i,key = None):
        '''
        get label of i-th data point as it is. 
        '''
        if key is None:
            return self.df.iloc[i][self.label_key]
        else:
            return self.df.iloc[i][key]
        
    def get_label_idx(self,i, key = None, target_transform = None,binary_label=False) :
        '''
        get label idx, which start from 0 incrementally
        self.target_transform is applied if exists
        '''
        label = self.get_label(i, key=key)
            
        if target_transform is None:
            target_transform = self.target_transform
            
        if target_transform is not None:
            if  isinstance(target_transform, dict):
                label_idx = target_transform[label]
            else:
                label_idx = target_transform(label)
        else:
            label_idx = int(label)
            
        return label_idx

    def __len__(self):
        return len(self.df)