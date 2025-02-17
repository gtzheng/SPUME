import torchvision
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import logging
import os
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.sampler import GroupTaskSampler, AttributesTaskSampler, RandomTaskSampler
import torch.nn.functional as F
import pickle
# import torch.multiprocessing as mp
import copy
from data.dataloader import get_loader

from torch.utils.tensorboard import SummaryWriter

import utils 
from methods import ERMModel, get_correlated_features, REPModel
from test import test_model, test_model_pseudo
import yaml

def meta_train(
    model,
    train_loader,
    idx_train_loader,
    val_loader,
    test_loader,
    args
):
    """Train the model using the meta-learning strategy

    Args:
        model (torch.nn.Module): a prediction model.
        train_loader (torch.utils.data.DataLoader): a train dataloader.
        idx_train_loader (torch.utils.data.DataLoader): a train dataloader that also returns the indexes of the data.
        val_loader (torch.utils.data.DataLoader): a validation dataloader.
        test_loader (torch.utils.data.DataLoader): a test dataloader.
        args (argparse.Namespace): arguments.

    Returns:
        None
    """
    timer = utils.Timer()
    logger = logging.getLogger("expr")
   

    criterion = torch.nn.CrossEntropyLoss()
    best_worst_acc = 0
    best_worst_acc_psu = 0
    best_avg_acc = 0
    best_unbiased_pseudo = 0
    get_best = False
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.e-4
    )
    # set the learning rate scheduler
    if args.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-6)
    else:
        lr_scheduler = None
    if args.random_sampler:
        sampler = RandomTaskSampler(train_loader.dataset, args.num_supp, args.num_query, args.num_episode, args.task_num)
        metatrain_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=4,
        )
        loader_func = lambda x,y,z,w: metatrain_loader
    elif args.use_group_label:
        sampler = GroupTaskSampler(train_loader.dataset, args.num_supp, args.num_query, args.num_episode)
        metatrain_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=4,
        )
        loader_func = lambda x,y,z,w: metatrain_loader
    else:
        def loader_func(model, dataset, idx_loader, score_func):
            class_correlated_feas = get_correlated_features(model, idx_loader, score_func)
            sampler = AttributesTaskSampler(dataset, args.num_supp, args.num_query, args.num_episode, args.task_num, args.topk, class_correlated_feas)
            metatrain_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                pin_memory=True,
                num_workers=4,
            )
            return metatrain_loader
    records = {}
    probs = None
    tolerance = 0
    writer = SummaryWriter(log_dir=args.tensorboard_path)
    for epoch in range(args.num_epochs):
        weight_epoch = []
        ave_meters = {k:utils.AverageMeter() for k in ["loss_all", "acc_cls", "loss_cls", "acc_rep", "loss_rep"]}
        model.train()
        for all_data in tqdm(loader_func(model, train_loader.dataset, idx_train_loader, args.score_func), leave=False):
            data, y, g, p, g_psu = all_data

            logits, embeds = model(data.cuda(),get_fea=True)
           

            embeds = embeds.reshape(args.task_num, (args.num_supp+args.num_query)*args.n_classes, -1)
            support_embeds, query_embeds = embeds[:, 0:args.num_supp*args.n_classes], embeds[:, args.num_supp*args.n_classes:]
            centroids_batch = support_embeds.reshape(args.task_num, args.n_classes, args.num_supp,-1).mean(dim=-2)
            
            query_labels = torch.repeat_interleave(torch.arange(args.n_classes), args.num_query).repeat(args.task_num)
            rep_logits = args.temp*torch.matmul(F.normalize(query_embeds,dim=-1), F.normalize(centroids_batch,dim=-1).permute(0,2,1)).reshape(-1,args.n_classes) # (B,Q,D) (B, N, D)
            rep_acc = (torch.argmax(rep_logits,dim=-1) == query_labels.cuda()).sum() / len(query_labels)
            rep_losses = criterion(rep_logits, query_labels.cuda())
            ave_meters["loss_rep"].update(rep_losses.item())
            ave_meters["acc_rep"].update(rep_acc.item())
            loss = rep_losses
            ave_meters["loss_all"].update(loss.item())
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        model.init(idx_train_loader, False)
        if args.use_group_label:
            avg_acc, worst_acc, unbiased_acc = test_model(model, val_loader)
        else:
            avg_acc, worst_acc_psu,  pseudo_unbiased = test_model_pseudo(model, val_loader, args.val_threshold_num)
            if args.dataset != "nico":
                avg_acc, worst_acc, unbiased_acc = test_model(model, val_loader)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        if best_avg_acc < avg_acc:
            best_avg_acc = avg_acc
            if args.n_gpu == 1:
                torch.save(model.state_dict(), args.avg_model_path)
            else:
                torch.save(model.module.state_dict(), args.avg_model_path)
            
        if args.use_group_label:
            if best_worst_acc < worst_acc:
                best_worst_acc = worst_acc
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.worst_model_path)
                else:
                    torch.save(model.module.state_dict(), args.worst_model_path)
                get_best = True
            else:
                get_best = False
        else:
            if best_worst_acc_psu < worst_acc_psu:
                best_worst_acc_psu = worst_acc_psu
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.worst_pseudo_model_path)
                else:
                    torch.save(model.module.state_dict(), args.worst_pseudo_model_path)
            if best_unbiased_pseudo < pseudo_unbiased:
                best_unbiased_pseudo = pseudo_unbiased
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.unbiased_pseudo_model_path)
                else:
                    torch.save(model.module.state_dict(), args.unbiased_pseudo_model_path)
                get_best = True
            else:
                get_best = False

            if args.dataset == "waterbirds" or args.dataset == "celeba":
                if best_worst_acc < worst_acc:
                    best_worst_acc = worst_acc
                    if args.n_gpu == 1:
                        torch.save(model.state_dict(), args.worst_model_path)
                    else:
                        torch.save(model.module.state_dict(), args.worst_model_path)
        train_loss_all = ave_meters["loss_all"].avg
        train_loss_rep = ave_meters["loss_rep"].avg
        train_loss_cls = ave_meters["loss_cls"].avg
        train_acc_rep = ave_meters["acc_rep"].avg
        train_acc_cls = ave_meters["acc_cls"].avg


        writer.add_scalar("Loss/train_all", train_loss_all, epoch)
        writer.add_scalar("Loss/train_rep", train_loss_rep, epoch)
        writer.add_scalar("Loss/train_cls", train_loss_cls, epoch)
        writer.add_scalar("Acc/train_cls", train_acc_cls, epoch)
        writer.add_scalar("Acc/train_rep", train_acc_rep, epoch)
        

        elapsed_time = timer.t()
        avg_time_per_epoch = elapsed_time / (epoch+1)
        if args.use_group_label:
            msg = (f"[{epoch}] Loss:{train_loss_all:.4f} " 
                    f"(CLS:{train_loss_cls:.4f}, " 
                    f"REP:{train_loss_rep:.4f}), "
                    f"CLSAcc:{train_acc_cls:.4f}, "
                    f"REPAcc:{train_acc_rep:.4f}, "
                    f"ValAcc: {avg_acc:.4f}, "
                    f"ValWAcc: {worst_acc:.4f}, "
                    f"ValUAcc: {unbiased_acc:.4f},"
                    f"TestAcc: {test_avg_acc:.4f}, "
                    f"TestWAcc: {test_worst_acc:.4f}, "
                    f"TestUAcc: {test_unbiased_acc:.4f}"
                    )
            writer.add_scalar("Acc/ValAcc", avg_acc, epoch)
            writer.add_scalar("Acc/ValWAcc", worst_acc, epoch)
            writer.add_scalar("Acc/ValUAcc", unbiased_acc, epoch)
            writer.add_scalar("Acc/TestAcc", test_avg_acc, epoch)
            writer.add_scalar("Acc/TestWAcc", test_worst_acc, epoch)
            writer.add_scalar("Acc/TestUAcc", test_unbiased_acc, epoch)
            # if get_best:
            #     msg += "(best)"
        else:
            msg = (f"[{epoch}] Loss:{train_loss_all:.4f} "
                    f"(CLS:{train_loss_cls:.4f}, "
                    f"REP:{train_loss_rep:.4f}), "
                    f"CLSAcc:{train_acc_cls:.4f}, "
                    f"REPAcc:{train_acc_rep:.4f}, "
                    f"ValAcc: {avg_acc:.4f}, "
            )
            if args.dataset != "nico":
                msg += f"ValUAcc: {unbiased_acc:.4f}, "
                msg += f"ValWAcc: {worst_acc:.4f}, "
            msg += (f"ValPWAcc: {worst_acc_psu:.4f}, "
                    f"ValPUAcc: {pseudo_unbiased:.4f}, "
                    f"TestAcc: {test_avg_acc:.4f}, "
                    f"TestWAcc: {test_worst_acc:.4f}, "
                    f"TestUAcc: {test_unbiased_acc:.4f}"
            )
            
            writer.add_scalar("Acc/ValAcc", avg_acc, epoch)
            writer.add_scalar("Acc/ValPWAcc", worst_acc_psu, epoch)
            writer.add_scalar("Acc/ValPUAcc", pseudo_unbiased, epoch)
            writer.add_scalar("Acc/TestAcc", test_avg_acc, epoch)
            writer.add_scalar("Acc/TestWAcc", test_worst_acc, epoch)
            writer.add_scalar("Acc/TestUAcc", test_unbiased_acc, epoch)
            
        msg += f" ({utils.time_str(elapsed_time)}/{utils.time_str(avg_time_per_epoch*args.num_epochs)})"
        logger.debug(msg)
        writer.flush()
        if lr_scheduler:
            lr_scheduler.step()
    writer.close()
       




if __name__ == "__main__":
    args = utils.get_config()
    if args.use_val: # choose whether to use valiation data for training
        data_tag = "train_and_val"
    else:
        data_tag = "train"
    if args.random_sampler:
        group_str = "random_sampler"
    elif args.use_group_label:
        group_str = "group_labels"
    else:
        group_str = args.vlm
    if args.pretrained:
        model_tag = "imagenet_weights"
    else:
        model_tag = "scratch"
        

    expr_name = f"{args.dataset}_{args.backbone}_lr_{args.lr:.6f}_{data_tag}_{args.batch_size}B_{args.topk}topK_{args.num_supp}ns_{args.num_query}nq_{args.num_epochs}Epochs_{args.num_episode}Epi_{args.task_num}T_{group_str}_{args.temp:.2f}temp_{args.alpha}alpha_{args.score_func}_{args.scheduler}_scheduler_{model_tag}_{args.tag}"
    expr_folder = os.path.join(args.save_folder, expr_name)
    os.makedirs(expr_folder, exist_ok=True)
    with open(os.path.join(expr_folder,"config.yaml"),"w") as f:
        yaml.dump(args, f)
    
    args.worst_model_path = os.path.join(expr_folder, "worst_model.pt")
    args.worst_pseudo_model_path = os.path.join(expr_folder, "pseudo_worst_model.pt")
    args.avg_model_path = os.path.join(expr_folder, "avg_model_embed.pt")
    args.unbiased_pseudo_model_path = os.path.join(expr_folder, "pseudo_unbiased_model.pt")
    args.tensorboard_path = os.path.join(expr_folder, "tensorboard")
    EXPR_LOG_PATH = os.path.join(expr_folder, "expr_train.log")

    

    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:args.n_gpu]])
    utils.set_gpu(gpu)
    os.makedirs(
        expr_folder,
        exist_ok=True,
    )
    logger = logging.getLogger("expr")
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(EXPR_LOG_PATH)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    train_loader, idx_train_loader, val_loader, test_loader = get_loader(args)
    model = REPModel(args.backbone, train_loader.dataset.n_classes, args.pretrained)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.cuda()
    if args.n_gpu == 1:
        model.init(idx_train_loader, False)
    else:
        model.module.init(idx_train_loader, False)
    if not args.test:
        meta_train(
            model,
            train_loader,
            idx_train_loader,
            val_loader,
            test_loader,
            args
        )
    

    state_dict = torch.load(args.avg_model_path)
    model.load_state_dict(state_dict)
    if args.n_gpu == 1:
        model.init(idx_train_loader)
    else:
        model.module.init(idx_train_loader)
    test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

    logger.info(f"[AvgModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}")
    if args.use_group_label or (args.dataset == "waterbirds" or args.dataset == "celeba"):
        state_dict = torch.load(args.worst_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            model.init(idx_train_loader)
        else:
            model.module.init(idx_train_loader)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(f"[WorstModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}")
    
    if not args.use_group_label:
        state_dict = torch.load(args.worst_pseudo_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            model.init(idx_train_loader)
        else:
            model.module.init(idx_train_loader)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(f"[PseudoWorstModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}")

        state_dict = torch.load(args.unbiased_pseudo_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            model.init(idx_train_loader)
        else:
            model.module.init(idx_train_loader)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(f"[PseudoUnbiasedModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}")


        
    
