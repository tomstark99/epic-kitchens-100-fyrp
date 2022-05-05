import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import h5py
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

class HDF5Dataset(Dataset):

    class LabelDataset(Dataset):

        def __init__(
            self,
            narration_ids: List[str],
            labels: List[Dict[str, Any]]
        ) -> None:
            self.narration_ids = narration_ids
            self.labels = labels
        def __len__(self) -> int:
            return len(self.labels)
        def __getitem__(self, key: int) -> Tuple[str, Dict[str, Any]]:
            return (self.narration_ids[key], self.labels[key])

    def __init__(
        self,
        hdf5_path: Path,
        pkl_path: Path
    ) -> None:
        self.hdf5_path = hdf5_path
        self.pkl_path = pkl_path
        self.hdf5_features: List[np.ndarray]
        self.label_dataset: self.LabelDataset
        self._load()

    def _load(self) -> None:
        with h5py.File(self.hdf5_path, 'r') as f_1:
            self.hdf5_features = {narr_id: np.array(features) for narr_id, features in f_1['visual_features'].items()}
        with open(self.pkl_path, 'rb') as f_2:
            pkl = pickle.load(f_2)
            self.label_dataset = self.LabelDataset(pkl['narration_id'], pkl['labels'])

    def __len__(self) -> int:
        return len(self.hdf5_features)

    def __getitem__(self, key: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        narr_id, label = self.label_dataset[key]
        features = self.hdf5_features[narr_id]
        return (features, {k: label[k] for k in ['narration_id', 'verb_class', 'noun_class']})

        
