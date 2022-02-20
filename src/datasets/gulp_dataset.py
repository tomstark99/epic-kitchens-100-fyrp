from typing import Dict, Any
from torch.utils.data import Dataset
from pathlib import Path
from gulpio2 import GulpDirectory

import numpy as np

class GulpDataset(Dataset):

    def __init__(self, gulp_dir: Path):
        self.narration_ids = []
        self.gulp_data: GulpDirectory
        self._load(gulp_dir)

    def _load(self, gulp_dir: Path):
        self.gulp_data = GulpDirectory(gulp_dir)
        self.narration_ids = list(self.gulp_data.merged_meta_dict.keys())
        
    def __len__(self):
        return len(self.narration_ids)

    def __getitem__(self, key: int):
        return self.gulp_data[self.narration_ids[key]]

class SubsetGulpDataset(Dataset):

    def __init__(self, gulp_dir: Path, subset_ids: List[str]):
        self.narration_ids = subset_ids
        self.gulp_data: GulpDirectory
        self._load(gulp_dir)

    def _load(self, gulp_dir: Path):
        self.gulp_data = GulpDirectory(gulp_dir)
        
    def __len__(self):
        return len(self.narration_ids)

    def __getitem__(self, key: int):
        return self.gulp_data[self.narration_ids[key]]
