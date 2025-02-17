import torch
import torch.nn as nn
import numpy as np
import os
import copy
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans

class GroupTaskSampler:
    def __init__(self, dataset, num_supp, num_query, num_batches, num_episodes):
        """Initialize a group task sampler

        Args:
            dataset (torch.utils.data.Dataset): a dataset.
            num_supp (int): number of support samples.
            num_query (int): number of query samples.
            num_batches (int): number of batches per epoch.
            num_episodes (int): number of episodes (tasks) per batch.
        """
        self.num_batches = num_batches
        self.y_array = dataset.y_array
        self.confounders = np.unique(dataset.confounder_array)
        self.classes = np.unique(self.y_array)
        self.confounder_array = dataset.confounder_array
        self.num_supp = num_supp
        self.num_query = num_query
        self.num_episodes = num_episodes
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        """Sample a batch of tasks based on the group labels.
        In each task, the support samples from each class are sampled from the same group, while the query samples are sampled from different groups.

        Raises:
            ValueError: If the number of support samples is smaller than the required number of support samples.
            ValueError: if the number of query samples is smaller than the required number of query samples.

        Yields:
            torch.tensor: a batch sample indexes.
        """
        for n in range(self.num_batches):
            batches = []
            for b in range(self.num_episodes):
                p_arr = np.random.choice(self.confounders, len(self.classes), replace=False)
                indexes = np.arange(len(self.y_array))
                group_cls_labels = []
                group_indexes = []
                for p in p_arr:
                    group_cls_labels.append(self.y_array[self.confounder_array==p])
                    group_indexes.append(indexes[self.confounder_array==p])
                
                group_cls_indexes = []
                for i in range(len(p_arr)):
                    group_cls_indexes.append([group_indexes[i][group_cls_labels[i]==c] for c in self.classes])
                
                supp_indexes = []
                for i in range(len(p_arr)):
                    num_s = len(group_cls_indexes[i][i])
                    if num_s < self.num_supp:
                        raise ValueError(f"too large num_supp: Actual {num_s}; Required: {self.num_supp}")
                    indexes = np.random.permutation(len(group_cls_indexes[i][i]))[0:self.num_supp]
                    supp_indexes.append(group_cls_indexes[i][i][indexes])
                
                query_indexes = []
                for i in range(len(p_arr)):
                    clist = list(np.arange(len(p_arr)))
                    clist.pop(i)
                    c = np.random.choice(clist)
                    num_q = len(group_cls_indexes[c][i])
                    if num_q < self.num_query:
                        raise ValueError(f"too large num_query: Actual {num_q}; Required: {self.num_query}")
                    indexes = np.random.permutation(len(group_cls_indexes[c][i]))[0:self.num_query]
                    query_indexes.append(group_cls_indexes[c][i][indexes])

                supp_indexes = np.concatenate(supp_indexes)
                query_indexes = np.concatenate(query_indexes)
                batch = np.concatenate([supp_indexes,query_indexes])
                batches.append(batch)
            batches = np.concatenate(batches)
            yield torch.tensor(batches)
            

class AttributesTaskSampler:
    def __init__(self, dataset, num_supp, num_query, num_batches, task_num, topk, class_correlated_feas):
        """Initialize an attribute task sampler.

        Args:
            dataset (torch.utils.data.Dataset): a dataset.
            num_supp (int): number of support samples.
            num_query (int): number of query samples.
            num_batches (int): number of batches per epoch.
            task_num (int): number of tasks per batch.
            topk (int): number of top-k attributes.
            class_correlated_feas (dict[int, tuple[np.array, np.array]]): a dictionary of spuriousness scores and the indexes of active features for each class.
        """
        self.num_batches = num_batches
        self.y_array = dataset.y_array
        self.embeddings = dataset.embeddings
        self.num_attrs = self.embeddings.shape[1]
        self.classes = np.unique(self.y_array)
        self.num_supp = num_supp
        self.num_query = num_query
        self.task_num = task_num
        self.class_correlated_feas = class_correlated_feas
        self.topk = topk

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        """Sample a batch of tasks based on the extracted attributes.
        In each task, the support samples from each class are sampled from the same group, while the query samples are sampled from different groups.


        Yields:
            torch.tensor: a batch sample indexes.
        """
        for n in range(self.num_batches):
            batches = []
            
            for b in range(self.task_num):
                supp_indexes = []
                query_indexes = []
                for c in self.classes:
                    indexes = np.arange(len(self.y_array))[self.y_array==c]
                    prob_pos = self.class_correlated_feas[c][0] - self.class_correlated_feas[c][0].min() + 1e-10
                    prob_pos = prob_pos/prob_pos.sum()

                    while True:
                        attr_support_indexes = np.random.choice(len(self.class_correlated_feas[c][1]), 2, p=prob_pos, replace=False)
                        attrs = self.class_correlated_feas[c][1][attr_support_indexes]
                    
                        attr_support, attr_query = attrs[0:1], attrs[1:2]

                        supp_all_indexes = indexes[self.embeddings[self.y_array==c][:,attr_support].sum(axis=1)> 0]
                        query_all_indexes = indexes[self.embeddings[self.y_array==c][:,attr_query].sum(axis=1)> 0]
                        supp_uni_indexes = list(set(supp_all_indexes)-set(query_all_indexes))
                        query_uni_indexes = list(set(query_all_indexes)-set(supp_all_indexes))
                        if len(supp_uni_indexes) > 0 and len(query_uni_indexes) > 0:
                            break

                    supp_sel_index = np.random.choice(supp_uni_indexes, self.num_supp, replace=True)
                    query_sel_index = np.random.choice(query_uni_indexes, self.num_query, replace=True)
                        
                    supp_indexes.append(supp_sel_index)
                    query_indexes.append(query_sel_index)
                supp_indexes = np.concatenate(supp_indexes)
                query_indexes = np.concatenate(query_indexes)
                batch = np.concatenate([supp_indexes,query_indexes])
                batches.append(batch)
            batches = np.concatenate(batches)
            yield torch.tensor(batches)
            

class RandomTaskSampler:
    def __init__(self, dataset, num_supp, num_query, num_batches, task_num, save_path=None):
        """Initialize a random task sampler.

        Args:
            dataset (torch.utils.data.Dataset): a dataset.
            num_supp (int): number of support samples.
            num_query (int): number of query samples.
            num_batches (int): number of batches per epoch.
            task_num (int): number of tasks per batch.
            save_path (str, optional): path to save the sampled tasks. Defaults to None.
        """
        self.num_batches = num_batches
        self.y_array = dataset.y_array
        self.classes = np.unique(self.y_array)
        self.num_supp = num_supp
        self.num_query = num_query
        self.task_num = task_num

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        """Return a batch of tasks sampled randomly.

        Yields:
            torch.tensor: a batch sample indexes.
        """
        for n in range(self.num_batches):
            batches = []
            for b in range(self.task_num):
                indexes = np.arange(len(self.y_array))
                supp_indexes = []
                query_indexes = []
                for l in self.classes:
                    sel_indexes = indexes[self.y_array == l]
                    idxes = np.random.permutation(len(sel_indexes))[0:self.num_supp+self.num_query]
                    supp_indexes.append(sel_indexes[idxes][0:self.num_supp])
                    query_indexes.append(sel_indexes[idxes][self.num_supp:])
                supp_indexes = np.concatenate(supp_indexes)
                query_indexes = np.concatenate(query_indexes)
                batch = np.concatenate([supp_indexes,query_indexes])
                batches.append(batch)
            batches = np.concatenate(batches)
            yield torch.tensor(batches)