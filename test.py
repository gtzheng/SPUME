import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import logging
import os
import torch.nn as nn
import copy
from utils import set_gpu, get_free_gpu
import utils
from data.dataloader import get_loader
from methods import ERMModel, REPModel
from pretrain import ERMCosineModel, prepare_model
import time
from models.resnet import resnet18, resnet50
from tqdm import tqdm


def test_model(model, loader):
    """Evaluate the model on the loader

    Args:
        model (torch.nn.Module): a prediction model.
        loader (torch.utils.data.DataLoader): a dataloader.

    Returns:
        float, float, float: the average accuracy, the worst accuracy, and the unbiased accuracy.
    """
    count = 0
    acc = 0
    model.eval()
    res = []
    groups = []
    with torch.no_grad():
        for x, y, g, p, _ in loader:
            x, y = (
                x.cuda(),
                y.cuda(),
            )
            out = model(x)
            pred = (torch.argmax(out, dim=-1) == y).detach().cpu().numpy()
            res.append(pred)
            groups.append(g.detach().cpu().numpy())
    res = np.concatenate(res)
    avg_acc = res.sum() / len(res)

    groups = np.concatenate(groups, axis=0)
    if groups.ndim == 1:
        num_group_types = 1
        groups = groups.reshape(-1, 1)
    else:
        num_group_types = groups.shape[1]
    unbiased_acc_avg = 0
    worst_acc_avg = 0

    for g_id in range(num_group_types):
        acc_group = []
        group_num = []

        unique_groups = np.unique(groups[:, g_id])
        group2idx = {g: i for i, g in enumerate(unique_groups)}
        for g in unique_groups:
            gres = res[groups[:, g_id] == g]
            if len(gres) < 10:
                continue
            acc_group.append(gres.sum() / len(gres))
            group_num.append(len(gres))
        acc_group = np.array(acc_group)
        unbiased_acc_avg += acc_group.mean()
        worst_acc_avg += acc_group.min()
    unbiased_acc_avg /= num_group_types
    worst_acc_avg /= num_group_types
    return avg_acc, worst_acc_avg, unbiased_acc_avg


def test_model_pseudo(model, loader, num_threshold=100):
    """Evaluate the model on the loader.

    Args:
        model (torch.nn.Module): a prediction model.
        loader (torch.utils.data.DataLoader): a dataloader.
        num_threshold (int, optional): the threshold of the number of samples in a pseudo-group formulated by the extracted concepts. This is used to remove too small groups where the model's performance is not representative. Defaults to 100.

    Returns:
        float, float, float: the average accuracy, the worst accuracy, and the unbiased accuracy.
    """
    count = 0
    acc = 0
    model.eval()
    res = []
    groups_psu = []
    with torch.no_grad():
        for x, y, _, p, g_arr in loader:
            x, y = (
                x.cuda(),
                y.cuda(),
            )
            out = model(x)
            pred = (torch.argmax(out, dim=-1) == y).detach().cpu().numpy()
            res.append(pred)
            groups_psu.append(g_arr.detach().cpu().numpy())
    groups_psu = np.concatenate(groups_psu)
    res = np.concatenate(res)

    attr_worst_acc = []
    attr_avg_acc = []
    for a in range(groups_psu.shape[1]):
        acc_group = []
        group_num = []
        groups = groups_psu[:, a]
        uni_groups = np.unique(groups)
        n_groups = len(uni_groups)
        for g in range(n_groups // 2, n_groups):
            gres = res[groups == g]
            if len(gres) > num_threshold:
                acc_group.append(gres.sum() / len(gres))
                group_num.append(len(gres))
        if len(acc_group) > 0:
            acc_group = np.array(acc_group)
            worst_acc_psu = acc_group.min()
            attr_worst_acc.append(worst_acc_psu)
            attr_avg_acc.append(acc_group)
    attr_worst_acc = np.array(attr_worst_acc)
    attr_avg_acc = np.concatenate(attr_avg_acc)
    avg_acc = res.sum() / len(res)

    return avg_acc, attr_worst_acc.min(), attr_avg_acc.mean()


if __name__ == "__main__":
    args = utils.get_config()
    train_loader, idx_train_loader, val_loader, test_loader = get_loader(args)

    # model = REPModel(args.backbone, train_loader.dataset.n_classes, args.pretrained)
    # model.cuda()
    # model.init(idx_train_loader)
    # model.load_state_dict(torch.load(args.ckpt))

    model = ERMCosineModel(args.backbone, 2, True)
    model.cuda()
    model.load_state_dict(torch.load(args.ckpt))

    avg_acc, worst_acc, unbiased_acc = test_model(model, val_loader)
    print(f"{avg_acc:.6f}, {worst_acc:.6f}, {unbiased_acc:.6f}")
    avg_acc, worst_acc, unbiased_acc = test_model(model, test_loader)
    print(f"{avg_acc:.6f}, {worst_acc:.6f}, {unbiased_acc:.6f}")
    # test_model_pseudo(model, test_loader, num_threshold=100)
