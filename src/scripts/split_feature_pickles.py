import argparse
import logging

from gulpio2 import GulpDirectory
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List

from ipdb import launch_ipdb_on_exception
from collections import Counter

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
parser.add_argument("features_pickle", type=Path, help="Path to pickle file to save features")
parser.add_argument("--num-workers", type=int, default=0, help="Number of features expected from frame")
parser.add_argument("--batch-size", type=int, default=128, help="Max frames to run through backbone 2D CNN at a time")
parser.add_argument("--feature-dim", type=int, default=256, help="Number of features expected from frame")

class PickleSplitter:

    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        self.length = 0
        self.narration_ids = []
        self.features = []
        self.labels = []
        self.load()
        self.video_ids = [n_id.split("_")[0] for n_id in self.narration_ids]
        self.video_frequencies = Counter(self.video_ids)
        self.unique_video_ids = set(self.video_ids)

    def save(self):
        for i, label in enumerate(self.labels):
            for k, v in label.items():
                if isinstance(label[k], list):
                    try:
                        self.labels[i][k] = v[0]
                    except Exception:
                        pass
        offset = 0
        for video_id in self.video_ids:
            try:
                new_offset = offset + self.video_frequencies[video_id]
                with open(self.pkl_path.parent / f'{video_id}_features.pkl', 'wb') as f:
                    pickle.dump({
                        self.video_frequencies[video_id],
                        self.narration_id[offset:new_offset],
                        self.features[:,offset:new_offset],
                        self.labels[offset:new_offset]
                    }, f)
                offset = new_offset
            except Exception as e:
                print(e)

    def load(self):
        try:
            with open(self.pkl_path, 'rb') as f:
                pkl_dict = pickle.load(f)
            self.length = pkl_dict['length']
            self.narration_ids = pkl_dict['narration_id']
            self.features = pkl_dict['features']
            self.labels = pkl_dict['labels']
        except FileNotFoundError:
            pass
def main(args):

    
    video_ids = 
