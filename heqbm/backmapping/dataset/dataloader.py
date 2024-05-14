import torch
from typing import List
from herobm.backmapping.dataset import Data


class Collater(object):

    def __init__(
            self,
            fixed_fields: List[str] = [],
            exclude_keys: List[str] = [],
        ):
            self.fixed_fields = fixed_fields
            self._exclude_keys = set(exclude_keys)

    @classmethod
    def for_dataset(
        cls,
        dataset,
        exclude_keys: List[str] = [],
    ):
        return cls(
            fixed_fields=getattr(dataset, "fixed_fields", []),
            exclude_keys=exclude_keys,
        )

    def collate(self, batch: List[Data]) -> Data:
        """Collate a list of data"""

        fields = batch[0].keys()
        out = Data({field: [] for field in fields})
        for b in batch:
            for field in fields:
                out[field].append(b[field])
        
        for field, vals in out.items():
            if field in self.fixed_fields:
                out[field] = vals[0]
            else:
                out[field] = torch.stack(vals, dim=0)
        
        return out

    def __call__(self, batch: List[Data]) -> Data:
        """Collate a list of data"""
        return self.collate(batch)
    
    @property
    def exclude_keys(self):
        return list(self._exclude_keys)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater.for_dataset(dataset, exclude_keys=exclude_keys),
            **kwargs,
        )