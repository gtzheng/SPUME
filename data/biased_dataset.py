import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from tqdm import tqdm

# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
# celeba_metadata: https://github.com/PolinaKirichenko/deep_feature_reweighting

class IdxDataset(Dataset):
    def __init__(self, dataset):
        """Initialize an indexed dataset

        Args:
            dataset (torch.utils.data.Dataset): a dataset
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return the data at the given index in the dataset along with the index

        Args:
            idx (int): index of the data

        Returns:
            tuple[int, Any]: the index and the data
        """
        return (idx, *self.dataset[idx])


class BiasedDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, concept_embed=None):
        """Initialize a biased dataset (for Waterbirds and CelebA)

        Args:
            basedir (_type_): _description_
            split (str, optional): specify the split. Defaults to "train".
            transform (torchvision.transforms.Compose, optional): data transformations. Defaults to None.
            concept_embed (str, optional): path to the concept embeddings. Defaults to None.
        """
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        split_info = metadata_df["split"].values
        split_info = split_info[split_info != 2] # train and val
        print(len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.p_array = self.metadata_df["place"].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df["place"].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (
            self.y_array * self.n_places + self.confounder_array
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
        self.y_counts = (
            (
                torch.arange(self.n_classes).unsqueeze(1)
                == torch.from_numpy(self.y_array)
            )
            .sum(1)
            .float()
        )
        self.p_counts = (
            (torch.arange(self.n_places).unsqueeze(
                1) == torch.from_numpy(self.p_array))
            .sum(1)
            .float()
        )
        self.filename_array = self.metadata_df["img_filename"].values
        if concept_embed and os.path.exists(concept_embed) and split != "test":
            with open(concept_embed, "rb") as f:
                self.embeddings = pickle.load(f)
            self.embeddings = self.embeddings[split_info == split_i]
        else:
            self.embeddings = None

    def __len__(self):
        return len(self.filename_array)

    def get_group(self, idx):
        """Get the pseudo-group of the given index

        Args:
            idx (int): index of the data
        Returns:
            int: the pseudo-group of the data at the given index.
        """
        y = self.y_array[idx]
        g = (self.embeddings[idx] == 1) * self.n_classes + y
        return g

    def __getitem__(self, idx):
        """Return the data at the given index in the dataset

        Args:
            idx (int): index of the data.

        Returns:
            tuple[torch.tensor if self.transform is given else PIL.Image.Image, int, int, int, int]: the image, the label, the group, the place, and the pseudo-group.
        """
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.embeddings is None:
            return img, y, g, p, p
        else:
            return img, y, g, p, self.get_group(idx)


def get_transform_biased(target_resolution, train, augment_data):
    """Get the data transformation for the Waterbirds/CelebA dataset

    Args:
        target_resolution (list[int, int]): the target resolution of the image.
        train (bool): whether the data is for training.
        augment_data (bool): whether to augment the data.

    Returns:
        torchvision.transforms.Compose: the data transformation.
    """
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0] * scale),
                        int(target_resolution[1] * scale),
                    )
                ),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
    return transform
