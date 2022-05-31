import argparse
import pickle
import pandas as pd
import numpy as np

from datasets.gulp_dataset import GulpDataset, SubsetGulpDataset
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from extract_verb_noun_links import extract_verb_noun_links
from extract_verb_noun_links import unique_list

from compute_dummy_esvs import compute_esvs

parser = argparse.ArgumentParser(
    description="Extract verb-noun links from a given dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("gulp_dir", type=Path, help="Path to gulp directory")
parser.add_argument("esv_dir", type=Path, help="Path to esv directory")
parser.add_argument("save_directory", type=Path, help="Path to directory to save files")
parser.add_argument("min_frames", type=int, help="Min number of frames to calculate ESVs for")
parser.add_argument("max_frames", type=int, help="Max number of frames to calculate ESVs for")
parser.add_argument("uids_csv", type=Path, help="Path to subset csv")
parser.add_argument("verb_classes", type=Path, help="Path to verb classes csv")
parser.add_argument("noun_classes", type=Path, help="Path to noun classes csv")
parser.add_argument("--dummy_esvs", default=False, action='store_true', help="also extract dummy esvs")
parser.add_argument("--classes", type=bool, default=False, help="Extract as pure class numbers")
parser.add_argument("--narration-id", type=bool, default=False, help="Extract with noun as tuple with narration_id")

def main(args):

    verbs = pd.read_csv(args.verb_classes)
    nouns = pd.read_csv(args.noun_classes)

    if args.uids_csv:
        uids: np.ndarray = pd.read_csv(args.uids_csv, converters={"uid": str})[
            "uid"
        ].values
        dataset = SubsetGulpDataset(args.gulp_dir, uids)
    else:
        dataset = GulpDataset(args.gulp_dir)

    if args.dummy_esvs:
        compute_dummy_esvs(args, dataset)
    else:
        prune_esvs(args, uids)

    extract_links(args, dataset, verbs, nouns)

def compute_dummy_esvs(args, dataset):
    for n_frames in range(args.min_frames, args.max_frames+1):
        data_to_persist = compute_esvs(dataset, n_frames)
        with open(args.save_directory / 'esvs' / f'mtrn-esv-n_frames={n_frames}.pkl', 'wb') as f:
            pickle.dump(data_to_persist, f, protocol=pickle.HIGHEST_PROTOCOL)

def extract_links(args, dataset, verbs, nouns):
    
    print('extracting links...')
    verb_noun = {}
    verb_noun = extract_verb_noun_links(
        dataset,
        verbs, 
        nouns, 
        verb_noun,
        classes=False,
        narration=False)
    with open(args.save_directory / 'labels' / 'verb_noun.pkl', 'wb') as f:
        pickle.dump({
            verb: unique_list(verb_noun[verb]) for verb in verb_noun.keys()
        }, f)

    verb_noun_classes = {}
    verb_noun_classes = extract_verb_noun_links(
        dataset,
        verbs, 
        nouns, 
        verb_noun_classes,
        classes=True,
        narration=False)
    with open(args.save_directory / 'labels' / 'verb_noun_classes.pkl', 'wb') as f:
        pickle.dump({
            verb: unique_list(verb_noun_classes[verb]) for verb in verb_noun_classes.keys()
        }, f)

    verb_noun_classes_narration = {}
    verb_noun_classes_narration = extract_verb_noun_links(
        dataset,
        verbs, 
        nouns, 
        verb_noun_classes_narration,
        classes=True,
        narration=True)
    with open(args.save_directory / 'labels' / 'verb_noun_classes_narration.pkl', 'wb') as f:
        pickle.dump({
            verb: unique_list(verb_noun_classes_narration[verb]) for verb in verb_noun_classes_narration.keys()
        }, f)

def prune_esvs(args, subset):

    with open(args.esv_dir / 'f_train_mtrn-esv-min_frames=1-max_frames=8.pkl', 'rb') as f:
        esvs_training = pickle.load(f)

    indicies = np.array([x in subset.tolist() for x in esvs_training['uids'].tolist()])

    print('pruning...')
    for k, v in tqdm(esvs_training.items()):
        if type(v) == dict:
            pass
        elif type(v) == list:
            new_x = []
            for x in v:
                new_x.append(x[indicies])
            esvs_training[k] = new_x
        elif v.ndim == 2:
            esvs_training[k] = v[:,indicies]
        elif v.ndim == 1:
            esvs_training[k] = v[indicies]

    with open(args.save_directory / 'esvs' / 'subset_f_train_mtrn-esv-min_frames=1-max_frames=8.pkl', 'wb') as f:
        pickle.dump(esvs_training, f)

if __name__ == '__main__':
    main(parser.parse_args())
