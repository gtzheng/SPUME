from torch.utils.data import DataLoader, RandomSampler
from data.biased_dataset import BiasedDataset, get_transform_biased, IdxDataset
import torch
import numpy as np

from data.in9_data import (
    get_imagenet_transform,
    ImageNet9,
    ImageNetA,
)

from data.nico_data import (
    get_transform_nico,
    NICO_dataset,
    TRAINING_DIST,
)


def add_val_to_train(trainset, valset):
    """Add a half of the validation set to the training set.

    Args:
        trainset (torch.utils.data.Dataset): the training dataset.
        valset (torch.utils.data.Dataset): the validation dataset.

    Returns:
        torch.utils.data.Dataset, torch.utils.data.Dataset: the updated training (with a half of the original validation data) and validation (containing the remaining half of the original validation data) datasets.
    """
    val_indexes = torch.randperm(len(valset))
    train_indexes = torch.randperm(len(trainset))

    num_sel = len(valset) // 2

    trainset.y_array = np.concatenate(
        [
            trainset.y_array[train_indexes[0:num_sel]],
            valset.y_array[val_indexes[0:num_sel]],
        ]
    )
    trainset.p_array = np.concatenate(
        [
            trainset.p_array[train_indexes[0:num_sel]],
            valset.p_array[val_indexes[0:num_sel]],
        ]
    )
    trainset.group_array = np.concatenate(
        [
            trainset.group_array[train_indexes[0:num_sel]],
            valset.group_array[val_indexes[0:num_sel]],
        ]
    )
    trainset.confounder_array = np.concatenate(
        [
            trainset.confounder_array[train_indexes[0:num_sel]],
            valset.confounder_array[val_indexes[0:num_sel]],
        ]
    )
    trainset.filename_array = np.concatenate(
        [
            trainset.filename_array[train_indexes[0:num_sel]],
            valset.filename_array[val_indexes[0:num_sel]],
        ]
    )
    if trainset.embeddings is not None:
        trainset.embeddings = np.concatenate(
            [
                trainset.embeddings[train_indexes[0:num_sel]],
                valset.embeddings[val_indexes[0:num_sel]],
            ]
        )
    train_group_counts = [
        (trainset.group_array == g).sum().item() for g in range(trainset.n_groups)
    ]

    valset.y_array = valset.y_array[val_indexes[num_sel:]]
    valset.p_array = valset.p_array[val_indexes[num_sel:]]
    valset.group_array = valset.group_array[val_indexes[num_sel:]]
    valset.confounder_array = valset.confounder_array[val_indexes[num_sel:]]
    valset.filename_array = valset.filename_array[val_indexes[num_sel:]]
    if valset.embeddings is not None:
        valset.embeddings = valset.embeddings[val_indexes[num_sel:]]
    val_group_counts = [
        (valset.group_array == g).sum().item() for g in range(valset.n_groups)
    ]
    return trainset, valset


def get_biased_loader(
    data_folder,
    augmentation,
    batch_size,
    attr_embed_path=None,
    batch_sampler=None,
    num_workers=8,
    use_val=False,
    fast_train=0,
):
    """Get the dataloaders for the Waterbirds/CelebA dataset.

    Args:
        data_folder (str): path to the dataset.
        augmentation (bool): choose whether to augment the data.
        batch_size (int): batch size.
        attr_embed_path (str, optional): path to the attribute embeddings. Defaults to None.
        batch_sampler (torch.utils.data.BatchSampler, optional): a batch sampler. Defaults to None.
        num_workers (int, optional): number of workers. Defaults to 8.
        use_val (bool, optional): choose whether to use a half of the validation set. Defaults to False.
        fast_train (int, optional): choose whether to use a subset of the training set for faster training. Defaults to 0. If fast_train > 0, the training set will be subsampled to fast_train*batch_size.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: training, training with indexes, validation, and test dataloaders.
    """
    train_transform = get_transform_biased(
        target_resolution=(224, 224), train=True, augment_data=True
    )
    test_transform = get_transform_biased(
        target_resolution=(224, 224), train=False, augment_data=False
    )
    trainset = BiasedDataset(
        basedir=data_folder,
        split="train",
        transform=train_transform if augmentation else test_transform,
        concept_embed=attr_embed_path,
    )
    trainset_ref = BiasedDataset(
        basedir=data_folder,
        split="train",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)

    valset = BiasedDataset(
        basedir=data_folder,
        split="val",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )

    testset = BiasedDataset(
        basedir=data_folder,
        split="test",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )
    if use_val:
        trainset, valset = add_val_to_train(trainset, valset)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if batch_sampler:
        train_loader = DataLoader(
            trainset, batch_sampler=batch_sampler, **loader_kwargs
        )
    else:
        if fast_train > 0 and len(trainset) > batch_size * fast_train:
            train_loader = DataLoader(
                trainset,
                sampler=RandomSampler(trainset, num_samples=batch_size * fast_train),
                pin_memory=True,
                num_workers=num_workers,
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                sampler=RandomSampler(
                    train_idx_dataset, num_samples=batch_size * fast_train
                ),
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                trainset, shuffle=True, batch_size=batch_size, **loader_kwargs
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
            )
    val_loader = DataLoader(
        valset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        testset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )

    return train_loader, idx_train_loader, val_loader, test_loader


def get_imagenet9_loader(
    im9_data_folder,
    imA_data_folder,
    im9_cluster_file,
    augmentation,
    batch_size,
    attr_embed_path=None,
    batch_sampler=None,
    num_workers=8,
    use_val=False,
    fast_train=0,
):
    """Get the dataloaders for the ImageNet-9 dataset.

    Args:
        im9_data_folder (str): path to the ImageNet-9 dataset.
        imA_data_folder (str): path to the ImageNet-A dataset.
        im9_cluster_file (str): path to the cluster file.
        augmentation (bool): choose whether to augment the data.
        batch_size (int): batch size.
        attr_embed_path (str, optional): path to the attribute embeddings. Defaults to None.
        batch_sampler (torch.utils.data.BatchSampler, optional): a batch sampler. Defaults to None.
        num_workers (int, optional): number of workers. Defaults to 8.
        use_val (bool, optional): choose whether to use a half of the validation set. Defaults to False.
        fast_train (int, optional): choose whether to use a subset of the training set for faster training. Defaults to 0. If fast_train > 0, the training set will be subsampled to fast_train*batch_size.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: training, training with indexes, validation, and test dataloaders.
    """
    train_transform = get_imagenet_transform(train=True, augment_data=True)
    test_transform = get_imagenet_transform(train=False, augment_data=False)

    trainset = ImageNet9(
        basedir=im9_data_folder,
        split="train",
        transform=train_transform if augmentation else test_transform,
        concept_embed=attr_embed_path,
    )
    trainset_ref = ImageNet9(
        basedir=im9_data_folder,
        split="train",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)

    valset = ImageNet9(
        basedir=im9_data_folder,
        split="val",
        transform=test_transform,
        concept_embed=attr_embed_path,
        cluster_file=im9_cluster_file,
    )
    if use_val:
        trainset, valset = add_val_to_train(trainset, valset)

    testset = ImageNetA(
        basedir=imA_data_folder,
        transform=test_transform,
    )

    if use_val:
        trainset, valset = add_val_to_train(trainset, valset)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if batch_sampler:
        train_loader = DataLoader(
            trainset, batch_sampler=batch_sampler, **loader_kwargs
        )
    else:
        if fast_train > 0 and len(trainset) > batch_size * fast_train:
            train_loader = DataLoader(
                trainset,
                batch_size=batch_size,
                sampler=RandomSampler(trainset, num_samples=batch_size * fast_train),
                pin_memory=True,
                num_workers=num_workers,
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(trainset, num_samples=batch_size * fast_train),
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                trainset, shuffle=True, batch_size=batch_size, **loader_kwargs
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )

    val_loader = DataLoader(
        valset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        testset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_nico_loader(
    data_folder,
    augmentation,
    batch_size,
    attr_embed_path=None,
    batch_sampler=None,
    num_workers=8,
    use_val=False,
    fast_train=0,
):
    """Get the dataloaders for the NICO dataset.

    Args:
        data_folder (str): path to the NICO dataset.
        augmentation (bool): choose whether to augment the data.
        batch_size (int): batch size.
        attr_embed_path (str, optional): path to the attribute embeddings. Defaults to None.
        batch_sampler (torch.utils.data.BatchSampler, optional): a batch sampler. Defaults to None.
        num_workers (int, optional): number of workers. Defaults to 8.
        use_val (bool, optional): choose whether to use a half of the validation set. Defaults to False.
        fast_train (int, optional): choose whether to use a subset of the training set for faster training. Defaults to 0. If fast_train > 0, the training set will be subsampled to fast_train*batch_size.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: training, training with indexes, validation, and test dataloaders.
    """
    train_transform = get_transform_nico(train=True, augment_data=True)
    test_transform = get_transform_nico(train=False, augment_data=False)

    trainset = NICO_dataset(
        basedir=data_folder,
        split="train",
        balance_factor=1.0,
        training_dist=TRAINING_DIST,
        transform=train_transform if augmentation else test_transform,
        concept_embed=attr_embed_path,
    )
    trainset_ref = NICO_dataset(
        basedir=data_folder,
        split="train",
        balance_factor=1.0,
        training_dist=TRAINING_DIST,
        transform=test_transform,
        concept_embed=attr_embed_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    valset = NICO_dataset(
        basedir=data_folder,
        split="val",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )
    testset = NICO_dataset(
        basedir=data_folder,
        split="test",
        transform=test_transform,
        concept_embed=attr_embed_path,
    )

    if use_val:
        trainset, valset = add_val_to_train(trainset, valset)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if batch_sampler:
        train_loader = DataLoader(
            trainset, batch_sampler=batch_sampler, **loader_kwargs
        )
    else:
        if fast_train > 0 and len(trainset) > batch_size * fast_train:
            train_loader = DataLoader(
                trainset,
                batch_size=batch_size,
                sampler=RandomSampler(trainset, num_samples=batch_size * fast_train),
                pin_memory=True,
                num_workers=num_workers,
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(trainset, num_samples=batch_size * fast_train),
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                trainset, shuffle=True, batch_size=batch_size, **loader_kwargs
            )
            idx_train_loader = DataLoader(
                train_idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )

    val_loader = DataLoader(
        valset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        testset, shuffle=False, batch_size=batch_size, **loader_kwargs
    )

    return train_loader, idx_train_loader, val_loader, test_loader


def get_loader(args, batch_sampler=None):
    """Get data loaders for the specified dataset.

    Args:
        args (argparse.Namespace): arguments.
        batch_sampler (torch.utils.data.BatchSampler, optional): a batch sampler. Defaults to None.

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: training, training with indexes, validation, and test dataloaders.
    """
    if args.vlm == "vit-gpt2":
        attr_embed_path = args.vit_gpt2_attr_embed_path
    elif args.vlm == "blip":
        attr_embed_path = args.blip_attr_embed_path

    if args.dataset == "waterbirds" or args.dataset == "celeba":
        train_loader, idx_train_loader, val_loader, test_loader = get_biased_loader(
            args.data_folder,
            args.augmentation,
            args.batch_size,
            attr_embed_path,
            batch_sampler,
            args.num_workers,
            args.use_val,
            args.fast_train,
        )
    if args.dataset == "imagenet-9":
        train_loader, idx_train_loader, val_loader, test_loader = get_imagenet9_loader(
            args.im9_data_folder,
            args.imA_data_folder,
            args.im9_cluster_file,
            args.augmentation,
            args.batch_size,
            attr_embed_path,
            batch_sampler,
            args.num_workers,
            args.use_val,
            args.fast_train,
        )
    if args.dataset == "nico":
        train_loader, idx_train_loader, val_loader, test_loader = get_nico_loader(
            args.data_folder,
            args.augmentation,
            args.batch_size,
            attr_embed_path,
            batch_sampler,
            args.num_workers,
            args.use_val,
            args.fast_train,
        )
    return train_loader, idx_train_loader, val_loader, test_loader
