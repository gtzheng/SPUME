import os
import sys
import re
import datetime

import numpy as np
import json
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pandas as pd
from PIL import Image
import pickle
import warnings

# https://github.com/Wangt-CN/CaaM
# https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing
TRAINING_DIST = {'dog': ['on_grass', 'in_water', 'in_cage', 'eating', 'on_beach', 'lying', 'running'],
                 'cat': ['on_snow', 'at_home', 'in_street', 'walking', 'in_river', 'in_cage', 'eating'],
                 'bear': ['in_forest', 'black', 'brown', 'eating_grass', 'in_water', 'lying', 'on_snow'],
                 'bird': ['on_ground', 'in_hand', 'on_branch', 'flying', 'eating', 'on_grass', 'standing'],
                 'cow': ['in_river', 'lying', 'standing', 'eating', 'in_forest', 'on_grass', 'on_snow'],
                 'elephant': ['in_zoo', 'in_circus', 'in_forest', 'in_river', 'eating', 'standing', 'on_grass'],
                 'horse': ['on_beach', 'aside_people', 'running', 'lying', 'on_grass', 'on_snow', 'in_forest'],
                 'monkey': ['sitting', 'walking', 'in_water', 'on_snow', 'in_forest', 'eating', 'on_grass'],
                 'rat': ['at_home', 'in_hole', 'in_cage', 'in_forest', 'in_water', 'on_grass', 'eating'],
                 'sheep': ['eating', 'on_road', 'walking', 'on_snow', 'on_grass', 'lying', 'in_forest']}


def prepare_metadata(NICO_DATA_FOLDER, NICO_CXT_DIC_PATH, NICO_CLASS_DIC_PATH):
    """Prepare metadata for the NICO dataset.

    Args:
        NICO_DATA_FOLDER (str): path to the NICO dataset folder.
        NICO_CXT_DIC_PATH (str): path to the context dictionary.
        NICO_CLASS_DIC_PATH (str): path to the class dictionary.
    """
    cxt_dic = json.load(open(NICO_CXT_DIC_PATH, 'r'))
    class_dic = json.load(open(NICO_CLASS_DIC_PATH, 'r'))
    cxt_index2name = {i: n for n, i in cxt_dic.items()}
    class_index2name = {i: n for n, i in class_dic.items()}

    labels = []
    contexts = []
    context_names = []
    label_names = []
    file_names = []
    splits = []
    for split_id, split in enumerate(["train", "val", "test"]):
        all_file_name = os.listdir(os.path.join(NICO_DATA_FOLDER, split))
        for file_name in all_file_name:
            label, context, index = file_name.split('_')
            file_names.append(os.path.join(split, file_name))
            contexts.append(int(context))
            context_names.append(cxt_index2name[int(context)])
            label_names.append(class_index2name[int(label)])
            labels.append(int(label))
            splits.append(split_id)

    labels_unique = sorted(list(set(labels)))
    contexts_unique = sorted(list(set(contexts)))
    label2unique = {l: i for i, l in enumerate(labels_unique)}
    context2unique = {c: i for i, c in enumerate(contexts_unique)}
    uniquelabel2name = {
        label2unique[l]: class_index2name[l] for l in labels_unique}
    uniquecontext2name = {
        context2unique[c]: cxt_index2name[c] for c in contexts_unique}

    name2uniquelabel = {n: l for l, n in uniquelabel2name.items()}
    name2uniquecontext = {n: c for c, n in uniquecontext2name.items()}

    with open(os.path.join(NICO_DATA_FOLDER, "metadata.csv"), "w") as f:
        f.write("img_id,img_filename,y,label_name,split,context,context_name\n")
        for i in range(len(file_names)):
            file_name = file_names[i]
            label = label2unique[labels[i]]
            label_name = label_names[i]
            split_id = splits[i]
            context = context2unique[contexts[i]]
            context_name = context_names[i]
            f.write(
                f"{i},{file_name},{label},{label_name},{split_id},{context},{context_name}\n")


def get_transform_nico(train, augment_data=True):
    """Get the transformation for the NICO dataset.

    Args:
        train (bool): whether the data is for training.
        augment_data (bool, optional): whether to augment the data. Defaults to True.

    Returns:
        torchvision.transforms.Compose: a composition of transformations.
    """
    mean = [0.52418953, 0.5233741, 0.44896784]
    std = [0.21851876, 0.2175944, 0.22552039]
    if train and augment_data:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class NICO_dataset(Dataset):
    def __init__(self, basedir, split, balance_factor=1.0, transform=None, training_dist=None, concept_embed=None):
        """Initialize the NICO dataset.

        Args:
            basedir (str): path to the dataset folder.
            split (str): split of the dataset.
            balance_factor (float, optional): not used. Defaults to 1.0.
            transform (torchvision.transforms.Compose, optional): dataset transform. Defaults to None.
            training_dist (dict[str,list[str]], optional): not used. Defaults to None.
            concept_embed (str, optional): path to the concept embeddings. Defaults to None.
        """
        super(NICO_dataset, self).__init__()
        assert split in ["train", "val", "test"], f"invalida split = {split}"
        self.basedir = basedir
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        split_info = metadata_df["split"].values
        split_info = split_info[split_info != 2]
        print(len(metadata_df))
        split_i = ["train", "val", "test"].index(split)
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        

        self.y_array = self.metadata_df["y"].values
        labelnames = self.metadata_df["label_name"].values
        self.labelname2index = {}
        for i in range(len(self.y_array)):
            self.labelname2index[labelnames[i]] = self.y_array[i]

        self.p_array = self.metadata_df["context"].values
        contextnames = self.metadata_df["context_name"].values
        self.contextname2index = {}
        for i in range(len(self.p_array)):
            self.contextname2index[contextnames[i]] = self.p_array[i]
        self.filename_array = self.metadata_df["img_filename"].values
        if balance_factor != 1:
            sel_indexes = self.reformulate_data_dist(
                balance_factor, training_dist)
            self.filename_array = self.filename_array[sel_indexes]
            self.y_array = self.y_array[sel_indexes]
            self.p_array = self.p_array[sel_indexes]
        print(len(self.y_array))
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.p_array).size

        self.group_array = (
            self.y_array * self.n_places + self.p_array
        ).astype("int")
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
            (
                torch.arange(self.n_groups).unsqueeze(1)
                == torch.from_numpy(self.group_array)
            )
            .sum(1)
            .float()
        )

        self.transform = transform

        if concept_embed and os.path.exists(concept_embed) and split != "test":
            with open(concept_embed, "rb") as f:
                self.embeddings = pickle.load(f)
            if balance_factor != 1:
                self.embeddings = self.embeddings[split_info ==
                                                  split_i][sel_indexes]
            else:
                self.embeddings = self.embeddings[split_info == split_i]
        else:
            self.embeddings = None

    def reformulate_data_dist(self, balance_factor, training_dist, seed=0):
        """Reformulate the data distribution.

        Args:
            balance_factor (float): control the balance of the data across different contexts.
            training_dist (dict[str, list[str]]): context distributions for each class.
            seed (int, optional): random seed. Defaults to 0.

        Returns:
            np.array: selected sample indexes.
        """
        sel_indexes = []
        np.random.seed(seed)
        for img_class in training_dist.keys():
            cls_num = len(training_dist[img_class])
            class_idx = np.where(np.array(self.y_array) ==
                                 self.labelname2index[img_class])[0]
            # img_class_labels = [self.y_array[idx] for idx in class_idx]
            # img_class_datas = [self.filename_array[idx] for idx in class_idx]
            img_class_contexts = [self.p_array[idx] for idx in class_idx]

            for index, img_context in enumerate(training_dist[img_class]):
                img_context_label = self.contextname2index[img_context]
                idx = np.where(np.array(img_class_contexts)
                               == img_context_label)[0]
                img_context_num = idx.shape[0]
                select_context_num = int(
                    img_context_num * (balance_factor**(index / (cls_num - 1.0))))
                np.random.shuffle(idx)

                selec_idx = idx[:select_context_num]
                sel_indexes.append(class_idx[selec_idx])
                # new_data.extend([img_class_datas[i] for i in selec_idx])
                # new_label.extend([img_class_labels[i] for i in selec_idx])
                # new_context.extend([img_class_contexts[i] for i in selec_idx])

        # self.filename_array = new_data
        # self.y_array = np.array(new_label)
        # self.p_array = np.array(new_context)
        sel_indexes = np.concatenate(sel_indexes)
        return sel_indexes

    def get_group(self, idx):
        """Get the pseudo-group of a sample.

        Args:
            idx (int): the sample index.

        Returns:
            int: the group of the sample.
        """
        y = self.y_array[idx]
        g = (self.embeddings[idx] == 1) * self.n_classes + y
        return g

    def __getitem__(self, idx):
        """Get the sample at the given index.

        Args:
            idx (int): the sample index.

        Returns:
            tuple: a tuple of the image, label, group, context, and pseudo-group of the sample
        """
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = self.y_array[idx]
        p = self.p_array[idx]
        g = self.group_array[idx]

        if self.embeddings is None:
            return img, y, g, p
        else:
            return img, y, g, p, self.get_group(idx)

    def __len__(self):
        return len(self.y_array)
