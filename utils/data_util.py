from dataset import StrokeDataset
from torch.utils.data import DataLoader
import os
import numpy as np


def load_data(opt, datasetType='train', permutation=False, shuffle=True):
    #sketch = load_dataset(opt.root)
    data_set = StrokeDataset(
        opt=opt,
        root=os.path.join('dataset',opt.dataset_name),
        dataset_name=opt.dataset_name,
        split=datasetType,
        permutation=permutation
    )
    data_loader = DataLoader(
        data_set,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers
    )
    return data_loader
