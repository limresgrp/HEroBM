import copy
import torch
import numpy as np

from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from heqbm.backmapping.dataset import Data
from heqbm.mapper import Mapper
from heqbm.utils import DataDict

def get_datasets_from_mapping(mapping: Mapper, conf: Dict, shuffle: bool = False) -> Tuple[Dict[str, np.ndarray]]:
    dataset: Dict = mapping.dataset
    
    n_frames = len(dataset[DataDict.ATOM_POSITION])
    n_train = conf.get("n_train", 0.8)
    n_valid = conf.get("n_valid", 0.1)
    if isinstance(n_train, float):
        n_train = int(n_frames * n_train)
    if isinstance(n_valid, float):
        n_valid = int(n_frames * n_valid)

    all_idcs = np.arange(0, n_frames)
    
    if shuffle:
        np.random.shuffle(all_idcs)
    
    train_idcs = all_idcs[:n_train]
    valid_idcs = all_idcs[n_train:n_train + n_valid]
    test_idcs = all_idcs[n_train + n_valid:]
    
    d_train, d_valid, d_test = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    
    dataset_keys = [
        DataDict.ATOM_POSITION, DataDict.BEAD_POSITION, DataDict.ATOM_FORCES,
    ]

    for k in dataset_keys:
        try:
            d_train[k] = d_train[k][train_idcs]
            d_valid[k] = d_valid[k][valid_idcs]
            d_test[k]  = d_test[k][test_idcs]
        except TypeError:
            print(f"{k} key is missing")
            continue
    
    d_train[DataDict.ORIGINAL_FRAMES_IDCS] = train_idcs
    d_valid[DataDict.ORIGINAL_FRAMES_IDCS] = valid_idcs
    d_test[DataDict.ORIGINAL_FRAMES_IDCS]  = test_idcs
    
    return d_train, d_valid, d_test, train_idcs, valid_idcs, test_idcs


class InMemoryDataset(Dataset):
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            fields: Optional[List[str]] = None,
            fixed_fields: List[str] = [
                DataDict.ATOM_TYPES,
                DataDict.BEAD_TYPES,
                DataDict.DIHEDRAL_IDCS,
                DataDict.BEAD2ATOM_IDCS,
                DataDict.BEAD2ATOM_IDCS_MASK,
                DataDict.BEAD_VERSOR_1_ATOM_IDCS,
                DataDict.BEAD_VERSOR_2_ATOM_IDCS,
                DataDict.OMEGA_DIH_IDCS,
            ],
        ):
        self.data = Data()
        self.fixed_fields = fixed_fields
        self.fields = None if fields is None else fields + self.fixed_fields

        assert DataDict.ATOM_POSITION in dataset.keys()

        for field, arr in dataset.items():
            if self.fields is not None and field not in self.fields:
                continue
            if not isinstance(arr, torch.Tensor):
                if not isinstance(arr, np.ndarray):
                    continue
                if np.issubdtype(arr.dtype, np.floating):
                    arr = torch.as_tensor(arr, dtype=torch.get_default_dtype())
                elif np.issubdtype(arr.dtype, np.integer):
                    arr = torch.as_tensor(arr, dtype=torch.long)
                elif np.issubdtype(arr.dtype, np.bool_):
                    arr = torch.as_tensor(arr, dtype=torch.bool)
                else:
                    continue
            self.data[field] = arr

    def __len__(self):
        return len(self.data[DataDict.ATOM_POSITION])

    def __getitem__(self, idx):
        return Data({k: v if k in self.fixed_fields else v[idx] for k, v in self.data.items()})