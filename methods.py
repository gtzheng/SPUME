import logging
import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn as nn
from models.resnet import resnet18, resnet50
import torchvision
import torch.nn.functional as F

def get_correlated_features(model, dataloader, score_func="tanh-abs-log"):
    """Calculate the spuriousness scores of the activated concepts for each class using the model, dataloader and score function.
    An activated concept for a class means that the concept is present in at least one image of that class.
    A spuriousness score for each activated concept reflects the discrepancy between prediction accuracies on images with and without the concept. A large score indicates a large discrepancy, which suggests that the concept is spurious.

    Args:
        model (torch.nn.Module): a prediction model.
        dataloader (torch.utils.data.DataLoader): a dataloader.
        score_func (str, optional): the type of the score function. Defaults to "tanh-abs-log".

    Returns:
        dict[int, tuple[np.array, np.array]]: a dictionary of spuriousness scores and the indexes of active features for each class.
    """
    class_wise_data = {}
    model.eval()
    with torch.no_grad():
        for idx, data, y, _, _, _ in tqdm(dataloader, leave=False):
            logits = model(data.cuda())
            logits = logits.detach().cpu()
            preds = torch.argmax(logits, dim=1).numpy()
            for i in range(len(y)):
                l = y[i].item()
                if l in class_wise_data:
                    class_wise_data[l].append((idx[i].item(),int(preds[i]==l)))
                else:
                    class_wise_data[l] = [(idx[i].item(),int(preds[i]==l))]
    embeddings = dataloader.dataset.dataset.embeddings  
    class_correlated_feas = {}
    
    eps = 1e-10
    for c in class_wise_data:
        count_pos = 0
        num_per_class = len(class_wise_data[c])
        counts_pos_w = np.zeros(embeddings.shape[1])
        counts_neg_w = np.zeros(embeddings.shape[1])
        
        counts_pos_wo = np.zeros(embeddings.shape[1])
        counts_neg_wo = np.zeros(embeddings.shape[1])
        for idx, pred_res in class_wise_data[c]:
            if pred_res == 1:
                counts_pos_w[embeddings[idx] == 1] += 1
                counts_pos_wo[embeddings[idx] != 1] += 1
                count_pos += 1
            else:
                counts_neg_w[embeddings[idx] == 1] += 1
                counts_neg_wo[embeddings[idx] != 1] += 1

        all_indexes = np.arange(embeddings.shape[1])
        active_indexes = all_indexes[(counts_pos_w + counts_neg_w) > 0]

        p_y1_w0 = counts_pos_wo[active_indexes] / (counts_pos_wo[active_indexes] + counts_neg_wo[active_indexes] + eps)
        p_y1_w1 = counts_pos_w[active_indexes] / (counts_pos_w[active_indexes] + counts_neg_w[active_indexes] + eps)

        # address the corner cases
        cond = (p_y1_w1 == 0) & (p_y1_w0 == 0)
        p_y1_w1[cond] = 1.0
        p_y1_w0[cond] = 1.0 
        
        if score_func == "tanh-abs-log":
            scores = np.tanh(abs(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps)))
        elif score_func == "tanh-log":
            scores = np.tanh(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps))
        elif score_func == "abs-log":
            scores = abs(np.log(p_y1_w1 / (p_y1_w0 + eps)+eps))
        elif score_func == "log":
            scores = np.log(p_y1_w1 / (p_y1_w0 + eps)+eps)
        elif score_func == "abs-diff":
            scores = abs(p_y1_w1 - p_y1_w0)
        elif score_func == "diff":
            scores = p_y1_w1 - p_y1_w0
        elif score_func == "exp-diff":
            scores = np.exp(p_y1_w1 - p_y1_w0)
        elif score_func == "exp-abs-diff":
            scores = np.exp(abs(p_y1_w1 - p_y1_w0))

        class_correlated_feas[c] = (scores, active_indexes)
    model.train()
    return class_correlated_feas



class ERMModel(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        """Initliaze an ERM model

        Args:
            backbone (str): specify the backbone of the model.
            num_classes (int): number of classes.
            pretrained (bool): whether to load the pretrained weights.
        """
        super(ERMModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(progress=True), strict=False)
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True), strict=False)
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim
        self.num_classes = num_classes
        self.fea_dim = d
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x, get_fea=False):
        """Calculate the logits (and embeddings of x if get_fea=True) of the model

        Args:
            x (torch.tensor): a batch of image tensors.
            get_fea (bool, optional): choose whether to also return feature embeddings. Defaults to False.

        Returns:
            [torch.tensor, Optional[torch.tensor]]: prediction logits and embeddings of x if get_fea=True.
        """
        fea = self.backbone(x)
        logits = self.fc(fea)
        if get_fea:
            return logits, fea
        else:
            return logits



class REPModel(nn.Module):
    def __init__(self, backbone, n_classes, pretrained):
        """Initialize a prediction model that uses a centroid classifier for predictions.
        This model is designed specifically for meta0-learning.

        Args:
            backbone (str): backbone of the model.
            n_classes (int): number of classes.
            pretrained (bool): choose whether to load the pretrained weights.
        """
        super(REPModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(progress=True), strict=False)
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True), strict=False)
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim
        self.n_classes = n_classes
        self.fea_dim = d
        # self.init(idx_dataloader, False)

    def init(self, idx_dataloader, use_all=True):
        """Initialize the centroids of the classifier using the dataloader.
        A centroid of a class is the average feature embedding of all the samples in that class.

        Args:
            idx_dataloader (torch.utils.data.DataLoader): a dataloader that also provides the indexes of the images.
            use_all (bool, optional): Choose whether to use all the samples in the dataset. If False, directly use idx_dataloader which can be created with a subset of the original dataset. Defaults to True.
        """
        self.centroids = torch.zeros(self.n_classes, self.fea_dim)
        self.counts = {c:0 for c in range(self.n_classes)}
        if use_all:
            dataloader = torch.utils.data.DataLoader(idx_dataloader.dataset, shuffle=False, batch_size=256, pin_memory=True, num_workers=12)
        else:
            dataloader = idx_dataloader
        self.eval()
        with torch.no_grad():
            for _, x, y, _, _, _ in tqdm(dataloader, desc="init classifier", leave=False):
                fea = self.backbone(x.cuda()).cpu()
                for i in range(len(y)):
                    self.centroids[y[i].item()] += fea[i]
                    self.counts[y[i].item()] += 1
            for c in self.counts:
                self.centroids[c] /= self.counts[c]
        self.centroids = self.centroids.cuda()

    def forward(self, x, get_fea=False):
        """Predict the logits of x using the centroid classifier.

        Args:
            x (torch.tensor): a batch of image tensors.
            get_fea (bool, optional): choose whether to return the image embeddings. Defaults to False.

        Returns:
            [torch.tensor, Optional[torch.tensor]]: prediction logits and embeddings of x if get_fea=True.
        """
        fea = self.backbone(x) # B, D
        logits = torch.matmul(F.normalize(fea, dim=-1), F.normalize(self.centroids, dim=-1).T)
        if get_fea:
            return logits, fea
        else:
            return logits
