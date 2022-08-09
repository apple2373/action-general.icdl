import os
import random


def temporal_batching_index(fr,length=16):
    '''
    Do padding or half-overlapping clips for video.
    
    Input:
        fr: number of frames
    Output:
        batch_indices: array for batch where each element is frame index 
    '''
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        return [[0]*left + list(range(fr)) + [fr-1]*right]
    
    batch_indices = []
    last_idx = fr - 1
    assert length%2 == 0
    half = int(length/2)
    for i in range(0,fr-half,half):
            frame_indices = [0,]*length
            for j in range(length):
                current_idx =  i + j 
                if current_idx < last_idx:
                    frame_indices[j] = current_idx
                else:
                    frame_indices[j] = last_idx
            batch_indices.append(frame_indices)
            
    return batch_indices

def temporal_sliding_window(clip,window = 16):
    '''
    Make a batched tensor with 16 frame sliding window with the overlap of 8. 
    If a clip is not the multiply of 8, it's padded with the last frames. (1,2...,13,14,14,14) for (1,..,14) 
    If a clip is less than 16 frames, padding is applied like (1,1,....,1,2,3,4,5,5,...,5,5) for (1,2,3,4,5)
    This can be used for sliding window evaluation.
    
    Input:  list of image paths
    Output: list of list of window-sized image paths
    '''

    batch_indices = temporal_batching_index(len(clip),length = window)
    
    return [[clip[idx] for idx in  indices] for indices in batch_indices]

def temporal_center_crop(clip,length = 16):
    '''
    Input:  list of image paths
    Output: list of selected image paths
    '''
    fr = len(clip) 
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        indicies =  [0]*left + list(range(fr)) + [fr-1]*right
        output =  [clip[i] for i in indicies]
    elif fr==length:
        output =  clip    
    else:
        middle = int(fr/2)
        assert length%2 == 0
        half = int(length/2)
        start = middle - half
        output =  clip[start : start+length]
        
    return output



def random_temporal_crop(clip,length = 16):
    '''
    Just randomly sample 16 consecutive frames
    if less than 16 frames, just add padding.
    '''
    fr = len(clip) 
    if fr < length: 
        #e.g. (1,2,3,4,5) to (1,1,....,1,2,3,4,5,5,...,5,5)
        right = int((length-fr)/2)
        left = length - right - fr
        indicies =  [0]*left + list(range(fr)) + [fr-1]*right
        output =  [clip[i] for i in indicies]
    elif fr==length:
        output =  clip
    else:
        start=random.randint(0,fr-length)
        output =  clip[start : start+length]
    return output


def use_all_frames(clip):
    '''
    Just use it as it is :)
    '''
    return clip


class TemporalTransform(object):
    def __init__(self,length,mode="center"):
        self.mode = mode
        self.length = length
        #pass dummpy in order to catch incoored mode
        self.__call__(range(128))
        
    def __call__(self, clip):
        if self.mode == "random":
            return random_temporal_crop(clip,self.length)
        elif self.mode == "center":
            return temporal_center_crop(clip,self.length)
        elif self.mode == "all" or self.mode == "nocrop":
            #note that length cannot be satisfied!
            return use_all_frames(clip)
        elif self.mode == "slide":
            #note that output has one more dimention
            return temporal_sliding_window(clip,self.length)
        else:
            raise NotImplementedError("this option is not defined:",self.mode)