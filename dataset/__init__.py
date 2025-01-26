import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.dataset import DGSM
from dataset.randaugment import RandomAugment


def create_dataset(config):
    train_dataset = DGSM(config=config, is_train=True)
    val_dataset = DGSM(config=config, is_train=False)
    return train_dataset, val_dataset


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = True #记得为False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders