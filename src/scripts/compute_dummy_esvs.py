import argparse
import logging

from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from torchvideo.samplers import frame_idx_to_list
from frame_sampling import RandomSampler

import torch
from datasets.gulp_dataset import GulpDataset

from omegaconf import OmegaConf

from systems import EpicActionRecognitionSystem
from systems import EpicActionRecogintionDataModule
from models.esvs import V_MTRN, N_MTRN

import pickle
import numpy as np
import pandas as pd

from attribution.online_shapley_value_attributor import OnlineShapleyAttributor
from subset_samplers import ConstructiveRandomSampler

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Compute dummy ESVs for n frames",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to features")
parser.add_argument("esvs_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--sample-n-frames", type=int, default=8, help="How many frames to sample to compute ESVs for")

def main(args):

    dataset = GulpDataset(args.gulp_dir)
    n_frames = args.sample_n_frames
    data_to_persist = compute_esvs(dataset, n_frames)

    with open(args.esvs_pickle, 'wb') as f:
            pickle.dump(data_to_persist, f, protocol=pickle.HIGHEST_PROTOCOL)

def compute_esvs(dataset: Dataset, n_frames: int):

    dataloader = DataLoader(dataset, batch_size=1)
    frame_sampler = RandomSampler(frame_count=n_frames, snippet_length=1, test=True)

    def subsample_frames(video_length: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        if video_length < n_frames:
            raise ValueError(f"Video too short to sample {n_frames} from")
        sample_idxs = np.array(frame_idx_to_list(frame_sampler.sample(video_length)))
        return sample_idxs

    data = {
        "labels": [],
        "uids": [],
        "sequence_idxs": [],
        "sequence_lengths": [],
        "scores": [],
        "shapley_values": [],
    }

    for i, (_, rgb_meta) in tqdm(
        enumerate(dataloader),
        unit=" action seq",
        total=len(dataloader),
        dynamic_ncols=True
    ):
        labels = {
            'verb': rgb_meta['verb_class'].item(),
            'noun': rgb_meta['noun_class'].item()
        }

        try:
            sample_idx = subsample_frames(rgb_meta['num_frames'].item())
        except ValueError:
            print(
                f"{uid} is too short ({rgb_meta['num_frames'].item()} frames) to sample {n_frames}"
                f"frames from."
            )
            continue

        result_scores = torch.softmax(torch.rand((1,397)), dim=-1)

        scores = {
            'verb': result_scores[:,:97].numpy(),#.cpu().numpy(),
            'noun': result_scores[:,97:].numpy()#.cpu().numpy()
        }
        result_esvs = torch.softmax(torch.rand((1,n_frames,397)), dim=-1)

        esvs = {
            'verb': result_esvs[:,:,:97].numpy(),
            'noun': result_esvs[:,:,97:].numpy()
        }
        
        rgb_meta['narration_id'] = rgb_meta['narration_id'][0]

        data["labels"].append(labels)
        data["uids"].append(rgb_meta['narration_id'])
        data["sequence_idxs"].append(sample_idx)
        data["sequence_lengths"].append(rgb_meta['num_frames'].item())#rgb_meta['num_frames'])
        data["scores"].append(scores)
        data["shapley_values"].append(esvs)
    
    def collate(vs: List[Any]):
        try:
            return np.stack(vs)
        except ValueError:
            return vs

    data_to_persist = {k: collate(vs) for k, vs in data.items()}

    return data_to_persist

    

if __name__ == "__main__":
    main(parser.parse_args())
