#system import
import os
import glob
import json
import sys
import copy
import random
import time
import math
        
#package import
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
#import cv2
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# from joblib import Parallel, delayed

#save pickle
import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        print(path,"saved")
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

#save json
def save_json(dict,path):
    with open(path, 'w') as f:
        json.dump(dict, f, sort_keys=True, indent=4)
        print(path,"saved")
        
def make_dir_if_needed(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

#utils
def make_cooccurrence(df,idx='verb_class_name',col="noun_class_name"):
    #make cooccurrence matrix
    verbs = sorted(list(df[idx].unique()))
    nouns = sorted(list(df[col].unique()))
    verb_noun_cooccur = np.zeros([len(verbs),len(nouns)])
    for i,row in tqdm(df.iterrows()):
        v=row[idx]
        n=row[col]
        v_idx = verbs.index(v)
        n_idx = nouns.index(n)
        verb_noun_cooccur[v_idx,n_idx] +=1
    df_coocur = pd.DataFrame(data=verb_noun_cooccur, index=verbs, columns=nouns, dtype=int)
    return df_coocur

def sort_cooccurrence(df,col=True,idx=True):
    idxs = df.index
    cols = df.columns
    if idx:
        idxs = df.index[np.argsort(df.T.sum()).values[::-1]]
    if col:
        cols = df.columns[np.argsort(df.sum()).values[::-1]]
    df = df.loc[idxs,cols] 
    return df

def get_common_unique(df_coocur):
    uniqe_objs = []
    common_objs = []
    verbs = df_coocur.index
    num_verbs = len(verbs)
    for noun,col in df_coocur.T.iterrows():
        num_nonzero = (col>0).sum()
        if num_nonzero==1:
            spike_idx = np.argmax(col>0)
            uniqe_objs.append((noun,verbs[spike_idx]))  
        elif num_nonzero==num_verbs:
            common_objs.append(noun)
        elif num_nonzero==0:
            pass
        else:
            assert False
    
    if len(uniqe_objs)==0:
        num_unique =0
    else:    
        df_uniqe_objs = pd.DataFrame(uniqe_objs,columns=["noun","verb"])
        df_uniqe_objs= df_uniqe_objs.groupby("verb").count()
        assert len(df_uniqe_objs.index) == num_verbs
        num_unique = 0
        if np.isnan(df_uniqe_objs["noun"].min()) and np.isnan(df_uniqe_objs["noun"].min()):
            num_unique = 0
        else:
            assert df_uniqe_objs["noun"].max() == df_uniqe_objs["noun"].min() 
            num_unique = df_uniqe_objs["noun"].max()

    return {"common":common_objs,"unique":uniqe_objs,"num_unique_nouns":num_unique,"num_common_nouns":len(common_objs)}

import iteround
def adjust_float_matrix(df):
    df = df.copy()
    #if all elemnets are already integter, just return
    if np.equal(np.mod(np.array(df), 1), 0).all():
        return df.astype(int)
    total = round(df.sum().sum())

    #https://pypi.org/project/iteround/
    #https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal/51451847#51451847
    data = np.array(df)
    shape = data.shape
    np.array(iteround.saferound(data.flatten(), 0)).reshape(shape)
    df.loc[:,:] = np.array(iteround.saferound(data.flatten(), 0)).reshape(shape) 
    assert total == df.sum().sum()
    return df.astype(int)

def setup_df(dataset_csv,exclude_missings,action_col,selected_verbs,obj_col,selected_nouns,num_cell_cap):
    df_selected = pd.read_csv(dataset_csv)
    df_selected =  df_selected.loc[df_selected[action_col].isin(selected_verbs)&df_selected[obj_col].isin(selected_nouns)].copy()
    if exclude_missings:
        print("excluding clips that is not included in something-else")
        assert missing_ids_json is not None
        assert os.path.exists(missing_ids_json)
        missing_ids = json.load(open(missing_ids_json))
        missing_ids = [int(intid) for intid in missing_ids]
        df_selected = df_selected.loc[~df_selected.id.isin(missing_ids)]
        
    df_cooccur = make_cooccurrence(df_selected,idx=action_col,col=obj_col)
    df_cooccur = df_cooccur.loc[selected_verbs,selected_nouns]
    if num_cell_cap == "median":
        num_cell_cap = int(np.median(np.array(df_cooccur).flatten()))
    print("num_cell_cap",num_cell_cap)
    return df_selected,df_cooccur,num_cell_cap

def construct_cooccurence(df_cooccur,df_cooccur_train,num_common,num_unique,total_num,num_cell_cap,min_num_test_cell,test_prop,time_out_sec=120,seed=0):    
    #this is just a rough estimate, as this has rounding operation.
    selected_verbs = df_cooccur_train.index
    num_actions = len(selected_verbs)
    num_per_cell_float = total_num/(num_actions*(num_common+num_unique))
    if num_per_cell_float < 1:
        print("total number too low")
        return None
    num_per_cell = math.ceil(num_per_cell_float)
    print("num_per_cell",num_per_cell)
    
    if num_common+num_actions*num_unique > len(df_cooccur_train.columns):
        print("not enough objects")
        return None
    
    #first, choose the common objects randomly
    #all cells has to have at least num_per_cell+min_num_test_cell
    common_objs = []
    if num_common>0:
        np.random.seed(seed)
        random.seed(seed)
        fail_flag = False
        min_count = num_per_cell+min_num_test_cell
        df_quality = (df_cooccur_train >= min_count).sum(axis=0) == num_actions
        candidate_objs = df_quality.index[df_quality]
        if len(candidate_objs) < num_common:
            print("failed, not enough samples")
            fail_flag  = True
            return None
        #common_objs = list(candidate_objs[np.random.permutation(len(candidate_objs))[0:num_common]])
        common_objs = list(candidate_objs[0:num_common])
        
    unique_action_obj_pairs = []
    if num_unique>0:
        #second, search the unique objects
        #as a preparation, let's check if there's enough samples or not
        def has_enough_samples(df_cooccur_train,action,obj,num_per_cell,min_num_test_cell):
            #given a aciton-objct pair,check if it has enough samples for training and test
            #check the two conditions
            #cond1: check the action paired with this object has the enough number of action-obj instances 
            cond1 = df_cooccur_train.loc[action,obj] >= num_per_cell+min_num_test_cell
            #cond2: check the other actions of this object column have the enough number of action-obj instances
            cond2 = (df_cooccur_train.loc[[ac for ac in df_cooccur_train.index if ac!=action],obj] >= min_num_test_cell).all()
            return cond1*cond2
        remaining_objs = [obj for obj in df_cooccur_train.columns if obj not in common_objs]
        remaining_actions = copy.copy(selected_verbs)
        df_enough_samples = df_cooccur_train.loc[selected_verbs,remaining_objs].copy()
        for action in selected_verbs:
            for obj in remaining_objs:
                #for each possible cell, check if it is selectable as unique obj
                df_enough_samples.loc[action,obj] =  has_enough_samples(df_cooccur_train,action,obj,num_per_cell,min_num_test_cell)

        #the algorithm is search in classical AI. it's actually depth first search
        def is_goal(df_coocur,num_unique_goal):
            uniqe_objs = []
            verbs = df_coocur.index
            num_verbs = len(verbs)
            for noun,col in df_coocur.T.iterrows():
                num_nonzero = (col>0).sum()
                if num_nonzero==1:
                    spike_idx = np.argmax(col>0)
                    uniqe_objs.append((noun,verbs[spike_idx]))  
            if len(uniqe_objs)==0:
                num_unique =0
            else:    
                df_uniqe_objs = pd.DataFrame(uniqe_objs,columns=["noun","verb"])
                df_uniqe_objs= df_uniqe_objs.groupby("verb").count()
                num_unique = 0
                if df_uniqe_objs["noun"].max() == df_uniqe_objs["noun"].min():
                    num_unique = df_uniqe_objs["noun"].max()
                if len(df_uniqe_objs.index) != num_verbs:
                    num_unique = -1
            return num_unique == num_unique_goal 

        def get_successors(df_cooccur_unique_subset,df_enough_samples,num_unique):
            next_possible_states = []
            remaining_actions = df_cooccur_unique_subset.sum(axis=1)<num_unique
            for obj,col in df_cooccur_unique_subset.T.iterrows():
                num_nonzero = (col>0).sum()
                if num_nonzero==0:
                    candidate_actions = df_enough_samples.index[remaining_actions*df_enough_samples.loc[:,obj]]
                    for action in candidate_actions:
                        state = df_cooccur_unique_subset.copy()
                        state.loc[action,obj] = 1
                        next_possible_states.append(state)
            return next_possible_states

        #depth first search
        np.random.seed(seed)
        random.seed(seed)
        fringe = []
        visited = set()
        #we call the each co-occurence matrix as a state
        state = pd.DataFrame(0, index=df_enough_samples.index, columns=df_enough_samples.columns)
        fringe.append(state)
        fail_flag = False
        start_time = time.time()
        while True:
            if (time.time() - start_time) > time_out_sec:
                fail_flag = True
                print("DFS time out",len(fringe))
                break
            if len(fringe) == 0:
                fail_flag = True
                print("DFS failed")
                break
            state = fringe.pop(-1)#depth first
            if is_goal(state,num_unique):
                print("DFS success")
                break
            visited.add(state.to_numpy().tobytes())
            successors = get_successors(state,df_enough_samples,num_unique)
            random.shuffle(successors)
            for state in successors: 
                if state.to_numpy().tobytes() not in visited:
                    fringe.append(state)
        if fail_flag:
            return None
        status = get_common_unique(state)
        unique_action_obj_pairs = get_common_unique(state)['unique']
        assert status['num_unique_nouns'] == num_unique

    #create co-occurence matrix for each
    df_coocur_train_subset = pd.DataFrame(index=df_cooccur.index,columns=df_cooccur.columns).fillna(0)
    df_coocur_test_subset = pd.DataFrame(index=df_cooccur.index,columns=df_cooccur.columns).fillna(0)
    df_coocur_val_subset = pd.DataFrame(index=df_cooccur.index,columns=df_cooccur.columns).fillna(0)
    df_test_type = pd.DataFrame(index=df_cooccur.index,columns=df_cooccur.columns)

    #take care of common obj
    for noun in common_objs:
        for verb in selected_verbs:
            #add train
            df_coocur_train_subset.loc[verb,noun] = num_per_cell_float
            num_remainings = df_cooccur.loc[verb,noun] - num_per_cell
            assert num_remainings>0
            #add test and val 
            #divide the num_remainings into test_prop and val_prop where test_prop+val_prop=1
            #but we cap the max into num_cell_cap
            num_test = min(num_cell_cap,round(test_prop*num_remainings))
            num_val = min(num_cell_cap,num_remainings-num_test)
            assert num_test+num_val+num_per_cell <= df_cooccur.loc[verb,noun]
            df_coocur_test_subset.loc[verb,noun] = num_test
            df_coocur_val_subset.loc[verb,noun] = num_val                        
            df_test_type.loc[verb,noun] = "testing_common_obj"

    #take care of unique_obj
    for noun,verb in unique_action_obj_pairs:
        #take care of this pair
        df_coocur_train_subset.loc[verb,noun] = num_per_cell_float
        num_remainings = df_cooccur.loc[verb,noun] - num_per_cell
        assert num_remainings>0
        num_test = min(num_cell_cap,round(test_prop*num_remainings))
        num_val = min(num_cell_cap,num_remainings-num_test)
        df_coocur_test_subset.loc[verb,noun] = num_test
        df_coocur_val_subset.loc[verb,noun] = num_val
        df_test_type.loc[verb,noun] = "testing_unqiue-self_obj"
        assert num_test+num_val+num_per_cell <= df_cooccur.loc[verb,noun]
        #take care of this noun but with other verbs
        paired_verb = verb
        for verb in selected_verbs:
            if verb == paired_verb:
                continue
            num_remainings = df_cooccur.loc[verb,noun]
            num_test = min(num_cell_cap,round(test_prop*num_remainings))
            num_val = min(num_cell_cap,num_remainings-num_test)
            assert num_test+num_val <= df_cooccur.loc[verb,noun]
            df_coocur_test_subset.loc[verb,noun] = num_test
            df_coocur_val_subset.loc[verb,noun] = num_val     
            df_test_type.loc[verb,noun] = "testing_unique-other_obj"
            assert num_test+num_val <= df_cooccur.loc[verb,noun]

    #take care of unseen obj
    for verb in selected_verbs:
        for noun in unseen_nouns:
            #add only val and test
            #divide the num_remainings into test_prop and val_prop where test_prop+val_prop=1
            #but we cap the max into num_cell_cap
            num_remainings = df_cooccur.loc[verb,noun]
            num_test = min(num_cell_cap,round(test_prop*num_remainings))
            num_val = min(num_cell_cap,num_remainings-num_test)
            assert num_test+num_val <= df_cooccur.loc[verb,noun]
            df_coocur_test_subset.loc[verb,noun] = num_test
            df_coocur_val_subset.loc[verb,noun] = num_val     
            df_test_type.loc[verb,noun] = "testing_unseen_obj"

    df_coocur_train_subset = adjust_float_matrix(df_coocur_train_subset)
    df_coocur_test_subset = adjust_float_matrix(df_coocur_test_subset)
    df_coocur_val_subset = adjust_float_matrix(df_coocur_val_subset)

    status = get_common_unique(df_coocur_train_subset)
    assert df_coocur_train_subset.sum().sum() == total_num
    assert status["num_common_nouns"] == num_common
    assert status["num_unique_nouns"] == num_unique

    name = "exp_N-%d_NumCommonNouns-%d_NumUniqueNouns-%d_seed-%d"%(total_num,num_common,num_unique,seed)
    configs = {"num_common_nouns":num_common,"num_unique_nouns":num_unique,"seed":seed,
               "time_out_sec":time_out_sec,"name":name,"N":total_num}
    setup = {}
    setup["name"] = name
    setup['configs'] = configs
    setup["train_coocur"] = df_coocur_train_subset
    setup["test_coocur"] = df_coocur_test_subset
    setup["val_coocur"] = df_coocur_val_subset
    setup['df_test_type'] = df_test_type

    assert (df_cooccur - (df_coocur_train_subset+df_coocur_test_subset+df_coocur_val_subset) >= 0).all().all()
    return setup

def sample_from_cooccur(df_all,df_cooccur,seed=0, allow_not_enough_samples=False):
    """
    Inputs:
        df_all: original data frame
        df_cooccur: a matrix of co-occurrence 
        seed: sampling seed
    Returns:
        df_sampled: sampled data frame from df_alll
    """
    df_sampled = []
    for verb,row in df_cooccur.iterrows():
        for noun,count in row.iteritems():
            df_segment = df_all.query("verb_class_name=='%s' and noun_class_name=='%s'"%(verb,noun))
            if allow_not_enough_samples and len(df_segment) < count:
                count = len(df_segment)
            assert count <= len(df_segment)
            df_segment = df_segment.sample(count,random_state=seed)
            df_sampled.append(df_segment)
    return  pd.concat(df_sampled)


def sampling_from_cooccurence(df_selected,setup,save_dir,samplingseed=100):
    save_name=setup["name"]
    df_coocur_train_subset = setup["train_coocur"]
    df_coocur_test_subset = setup["test_coocur"]
    df_coocur_val_subset = setup["val_coocur"]
    df_test_type = setup['df_test_type']
    configs = setup["configs"]
    total_num = configs["N"]

    df_selected_remain = df_selected.copy()
    df_this_train = sample_from_cooccur(df_selected_remain,df_coocur_train_subset,seed=samplingseed)
    df_selected_remain = df_selected_remain.loc[~df_selected_remain.index.isin(df_this_train.index)]
    df_this_test = sample_from_cooccur(df_selected_remain,df_coocur_test_subset,seed=samplingseed)
    df_selected_remain = df_selected_remain.loc[~df_selected_remain.index.isin(df_this_test.index)]
    df_this_val = sample_from_cooccur(df_selected_remain,df_coocur_val_subset,seed=samplingseed)

    #check if there's no overlap among train/val/test
    assert len(df_this_train.index.intersection(df_this_test.index))==0
    assert len(df_this_train.index.intersection(df_this_val.index))==0
    assert len(df_this_val.index.intersection(df_this_test.index))==0
    assert total_num == len(df_this_train)

    make_dir_if_needed(save_dir)
    print("saving",os.path.join(save_dir,save_name+".[train|val|test].csv"))
    df_this_train.to_csv(os.path.join(save_dir,save_name+".train.csv"))
    df_this_test.to_csv(os.path.join(save_dir,save_name+".test.csv"))
    df_this_val.to_csv(os.path.join(save_dir,save_name+".val.csv"))

if __name__=="__main__":
    print("making subset!")
    
    #TODO: make this settings into argument or something separated from this code
    ###
    # <settings> start
    ###
    dataset_csv = "./data/df_selected_cleaned.csv"
    index_name = "id"
    save_dir = "./data/cvpr-v4-rerun/"
    obj_col = "noun_class_name"
    action_col = "verb_class_name"
    selected_verbs = ['Picking [something] up', 'Putting [something]', 'Dropping [something]', 'Moving [something] from left to right', 'Moving [something] from right to left']
    selected_nouns = ['pencil', 'bottle', 'pen', 'book', 'box', 'remote', 'key', 'wallet', 'phone', 'marker', 'paper', 'charger', 'knife', 'battery', 'mouse', 'cup', 'watch', 'brush', 'glasses', 'scissors', 'comb', 'spoon', 'lighter', 'toy', 'tissue', 'shoe', 'screwdriver', 'stapler', 'glass', 'calculator']
    train_nouns = selected_nouns[0:20]
    unseen_nouns = selected_nouns[20:30]
    num_cell_cap = "median" #number to cap the maxinum number of action-object instances for evaluation. 'median' means median from the cells
    min_num_test_cell = 10 #minimum of the val and test instance we want to have
    test_prop = 0.8 #the test propotion, the val propotion will be 1 - test_prop
    missing_ids_json = "./data/missing_vids.json"
    seeds = range(10)
    time_out_sec = 60  
    parallel = True
    ###
    # </settings> done
    ###
   
    for exclude_missings in [False,True]:
        df_selected,df_cooccur,num_cell_cap = setup_df(dataset_csv,exclude_missings,action_col,selected_verbs,obj_col,selected_nouns,num_cell_cap)    
        df_cooccur_train = df_cooccur.loc[:,train_nouns]            
        def process(total_num,num_common,num_unique,seed):
            setup = construct_cooccurence(df_cooccur,df_cooccur_train,num_common,num_unique,total_num,num_cell_cap,min_num_test_cell,test_prop,time_out_sec=time_out_sec,seed=seed)
            if setup is not None:
                setup["configs"]["missingexcluded"] = exclude_missings
                if exclude_missings:
                    setup["name"] = setup["name"]+"_missingexcluded"
                make_dir_if_needed(save_dir)
                #CAUTION! this is assuming samplingseed = seed 
                setup["name"]=setup["name"]+"_samplingseed-%d"%seed
                save_pkl(os.path.join(save_dir,setup["name"]+".setup.pkl"),setup)
                sampling_from_cooccurence(df_selected,setup,save_dir,samplingseed=seed)

        iterations  = []
        for seed in seeds:
            for total_num in [375]:
                for num_common in [0,1,2,3,4,5]:
                    for num_unique in [0,1,2,3,4]:
                        if num_common==0 and num_unique==0:
                            continue
                        iterations.append((total_num,num_common,num_unique,seed)) 
            for total_num in [750,250,125]:
                for num_common in [1,2,3,4,5]:
                    num_unique = 0
                    iterations.append((total_num,num_common,num_unique,seed))
                
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1,verbose=10)( [delayed(process)(total_num,num_common,num_unique,seed) for total_num,num_common,num_unique,seed in iterations] )
        else:
            for total_num,num_common,num_unique,seed in iterations:
                process(total_num,num_common,num_unique,seed)