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
    def __init__(self, dataset, num_supp, num_query, num_batches, task_num, topk, class_correlated_feas, vocab_path=None):
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
        if vocab_path:
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            self.vocab = []

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for n in range(self.num_batches):
            batches = []
            
            for b in range(self.task_num):
                supp_indexes = []
                query_indexes = []
                for c in self.classes:
                    indexes = np.arange(len(self.y_array))[self.y_array==c]
                    prob_pos = self.class_correlated_feas[c][0] - self.class_correlated_feas[c][0].min() + 1e-10
                    prob_pos = prob_pos/prob_pos.sum()
                    
                    # prob_neg = -(self.class_correlated_feas[c][0] - self.class_correlated_feas[c][0].max()) + 1e-10
                    # prob_neg = prob_neg/prob_neg.sum()
                    while True:
                        attr_support_indexes = np.random.choice(len(self.class_correlated_feas[c][1]), 2, p=prob_pos, replace=False)
                        attrs = self.class_correlated_feas[c][1][attr_support_indexes]
                        # prob_neg[attr_support_indexes] = 0
                        # prob_neg = prob_neg / prob_neg.sum()
                        # attr_query = np.random.choice(self.class_correlated_feas[c][1], self.topk, p=prob_neg, replace=False)

                        attr_support, attr_query = attrs[0:1], attrs[1:2]

                        supp_all_indexes = indexes[self.embeddings[self.y_array==c][:,attr_support].sum(axis=1)> 0]
                        query_all_indexes = indexes[self.embeddings[self.y_array==c][:,attr_query].sum(axis=1)> 0]
                        supp_uni_indexes = list(set(supp_all_indexes)-set(query_all_indexes))
                        query_uni_indexes = list(set(query_all_indexes)-set(supp_all_indexes))
                        if len(supp_uni_indexes) > 0 and len(query_uni_indexes) > 0:
                            # if len(self.vocab)>0:
                            #     print(f"[T{b}]-{c} {self.vocab[attrs[0]]}, {self.vocab[attrs[1]]}")
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
            

           

class AttributeClusterTaskSampler:
    def __init__(self, dataset, num_supp, num_query, num_batches, task_num, topk, class_correlated_feas):
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

        all_active_indexes = []
        for c in class_correlated_feas:
            all_active_indexes.append(class_correlated_feas[c][1])
        all_active_indexes = np.unique(np.concatenate(all_active_indexes))
        ori2idx = {l:i for i,l in enumerate(all_active_indexes)}
        cls_indexes = {}
        for c in class_correlated_feas:
            cls_indexes[c] = np.array([ori2idx[l] for l in class_correlated_feas[c][1]])

        embeddings = dataset.embeddings[:,all_active_indexes]
        all_indexes = np.arange(len(dataset.y_array))
        self.support_indexes = {}
        self.query_indexes = {}
        self.cluster_indexes = {}
        for c in class_correlated_feas:
            cls_embeddings = embeddings[dataset.y_array==c][:, cls_indexes[c]] * (class_correlated_feas[c][0].reshape(1,-1))
            cls_indexes_arr = all_indexes[dataset.y_array==c]
            kmeans = KMeans(n_clusters=topk,n_init="auto").fit(cls_embeddings)
            self.cluster_indexes[c] = [cls_indexes_arr[kmeans.labels_==i] for i in range(topk)]
            # self.support_indexes[c] = cls_indexes_arr[kmeans.labels_==0]
            # self.query_indexes[c] = cls_indexes_arr[kmeans.labels_==1]
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for n in range(self.num_batches):
            batches = []
            
            for b in range(self.task_num):
                supp_indexes = []
                query_indexes = []
                
                for c in self.classes:
                    while True:
                        clusteres = np.random.choice(self.topk, 2, replace=False)
                        cand_supp_indexes = self.cluster_indexes[c][clusteres[0]]
                        cand_query_indexes = self.cluster_indexes[c][clusteres[1]]
                        if len(cand_supp_indexes) > 0 or len(cand_query_indexes) > 0:
                            break
                    if len(cand_supp_indexes) > 0 and len(cand_query_indexes) > 0:
                        supp_sel_index = np.random.choice(cand_supp_indexes, self.num_supp, replace=True)
                        query_sel_index = np.random.choice(cand_query_indexes, self.num_query, replace=True)
                    elif len(cand_supp_indexes) == 0:
                        sel_index = np.random.choice(cand_query_indexes, self.num_supp+self.num_query, replace=True)
                        supp_sel_index = sel_index[0:self.num_supp]
                        query_sel_index = sel_index[self.num_supp:]
                    elif len(cand_query_indexes) == 0:
                        sel_index = np.random.choice(cand_supp_indexes, self.num_supp+self.num_query, replace=True)
                        supp_sel_index = sel_index[0:self.num_supp]
                        query_sel_index = sel_index[self.num_supp:]
                   
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
        self.num_batches = num_batches
        self.y_array = dataset.y_array
        self.classes = np.unique(self.y_array)
        self.num_supp = num_supp
        self.num_query = num_query
        self.task_num = task_num

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
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