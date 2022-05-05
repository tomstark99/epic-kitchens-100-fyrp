import argparse
from itertools import dropwhile
import logging

from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from systems import EpicActionRecogintionShapleyClassifier

from models.esvs import V_MF, N_MF

from datasets.hdf5_dataset import HDF5Dataset
from frame_sampling import RandomSampler

from ipdb import launch_ipdb_on_exception
from tqdm import tqdm

import plotly.graph_objects as go
import numpy as np
import pickle

from livelossplot import PlotLosses

parser = argparse.ArgumentParser(
    description="Extract per-frame features from given dataset and backbone",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("features_hdf5", type=Path, help="Path to hdf5 file to save features")
parser.add_argument("features_pkl", type=Path, help="Path to pickle file to save features")
parser.add_argument("val_features_hdf5", type=Path, help="Path to validation features hdf5")
parser.add_argument("val_features_pkl", type=Path, help="Path to validation features pickle")
parser.add_argument("model_params_dir", type=Path, help="Path to save model parameters (not file name)")
parser.add_argument("--min-frames", type=int, default=1, help="min frames to train models for")
parser.add_argument("--max-frames", type=int, default=8, help="max frames to train models for")
parser.add_argument("--batch-size", type=int, default=512, help="mini-batch size of frame features to run through")
parser.add_argument("--epoch", type=int, default=200, help="How many epochs to do over the dataset")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for training optimiser")
parser.add_argument("--type", type=str, default='verb', help="Which class to train")
parser.add_argument("--hidden-layer-size", type=int, default=512, help="Hidden layer size")
parser.add_argument("--dropout-probability", type=float, default=0.5, help="Probability of dropouts")
# parser.add_argument("results_pkl", type=Path, help="Path to save training results")
# parser.add_argument("--test", type=bool, default=False, help="Set test mode to true or false on the RandomSampler")
# parser.add_argument("--log_interval", type=int, default=10, help="How many iterations between outputting running loss")
# parser.add_argument("--n_frames", type=int, help="Number of frames for 2D CNN backbone")
# parser.add_argument("--save_fig", type=Path, help="Save a graph showing lr / loss")

def no_collate(args):
    return args

def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    
    if args.type == 'verb':
        models = [V_MF(frame_count=i, hidden_layer_size=args.hidden_layer_size, dropout_probability=args.dropout_probability).to(device) for i in range(1,args.max_frames+1)]
        optimisers = [Adam(m.parameters(), lr=1e-5) for m in models]
    elif args.type == 'noun':
        models = [N_MF(frame_count=i, hidden_layer_size=args.hidden_layer_size, dropout_probability=args.dropout_probability).to(device) for i in range(1,args.max_frames+1)]
        optimisers = [Adam(m.parameters(), lr=1e-5) for m in models]
    else:
        raise ValueError(f"unknown type: {args.type}, known types are 'verb' and 'noun'")

    train_frame_samplers = [RandomSampler(frame_count=i, snippet_length=1, test=False) for i in range(1, args.max_frames+1)]
    test_frame_samplers = [RandomSampler(frame_count=i, snippet_length=1, test=True) for i in range(1, args.max_frames+1)]

    train(
        args,
        device,
        models,
        optimisers,
        train_frame_samplers,
        test_frame_samplers
    )

    # with open(args.results_pkl, 'wb') as f:
    #     pickle.dump(results, f)
    

def test():
    return 0

def train(
    args,
    device,
    models: List[nn.Module],
    optimisers: List[Adam],
    train_frame_samplers: List[RandomSampler],
    test_frame_samplers: List[RandomSampler]
):
    assert len(models) == len(optimisers)
    assert len(models) == len(train_frame_samplers)
    assert len(models) == len(test_frame_samplers)

    trainloader = DataLoader(HDF5Dataset(args.features_hdf5, args.features_pkl), batch_size=args.batch_size, collate_fn=no_collate, shuffle=True)
    testloader = DataLoader(HDF5Dataset(args.val_features_hdf5, args.val_features_pkl), batch_size=args.batch_size, collate_fn=no_collate, shuffle=False)

    writer = SummaryWriter(get_summary_writer_log_dir(args), flush_secs=1)

    training_result = []
    testing_result = []

    print(f"Training {args.type}s"f" for {args.min_frames}"f" - {args.max_frames}"f" frames...")

    for i in tqdm( # m, o, f
        # zip(models, optimisers, frame_samplers),
        range(args.min_frames-1, args.max_frames),
        unit=" model",
        dynamic_ncols=True
    ):
        classifier = EpicActionRecogintionShapleyClassifier(
            models[i],
            device,
            optimisers[i],
            train_frame_samplers[i],
            test_frame_samplers[i],
            trainloader,
            testloader,
            args.type
        )

        # model_train_results = {
        #     'running_loss': [],
        #     'running_acc1': [],
        #     'running_acc5': [],
        #     'epoch_loss': [],
        #     'epoch_acc1': [],
        #     'epoch_acc5': []
        # }
        # model_test_results = {
        #     'running_loss': [],
        #     'running_acc1': [],
        #     'running_acc5': [],
        #     'epoch_loss': [],
        #     'epoch_acc1': [],
        #     'epoch_acc5': []
        # }

        liveloss = PlotLosses()

        print(f'Training MF with model architecture:\n{classifier.model}\nlr: {args.learning_rate}\nbatch size: {args.batch_size}\nepochs: {args.epoch}')

        for epoch in tqdm(
            range(args.epoch),
            unit=" epoch",
            dynamic_ncols=True
        ):
            logs = {}

            train_result = classifier.train_step()

            epoch_loss = sum(train_result[f'{models[i].frame_count}_loss']) / len(trainloader)
            epoch_acc1 = sum(train_result[f'{models[i].frame_count}_acc1']) / len(trainloader)
            epoch_acc5 = sum(train_result[f'{models[i].frame_count}_acc5']) / len(trainloader)

            # model_train_results['running_loss'].append(train_result[f'{models[i].frame_count}_loss'])
            # model_train_results['running_acc1'].append(train_result[f'{models[i].frame_count}_acc1'])
            # model_train_results['running_acc5'].append(train_result[f'{models[i].frame_count}_acc5'])
            # model_train_results['epoch_loss'].append(epoch_loss)
            # model_train_results['epoch_acc1'].append(epoch_acc1)
            # model_train_results['epoch_acc5'].append(epoch_acc5)

            writer.add_scalar(f'training loss frames={models[i].frame_count}', epoch_loss, epoch)
            writer.add_scalars('combined training loss', {f'loss frames={models[i].frame_count}': epoch_loss}, epoch)
            writer.add_scalars(f'training accuracy frames={models[i].frame_count}', {'acc1': epoch_acc1, 'acc5': epoch_acc5}, epoch)
            writer.add_scalars('combined training accuracy', {f'acc1 frames={models[i].frame_count}': epoch_acc1, f'acc5 frames={models[i].frame_count}': epoch_acc5}, epoch)

            test_result = classifier.test_step()

            epoch_loss_ = sum(test_result[f'{models[i].frame_count}_loss']) / len(testloader)
            epoch_acc1_ = sum(test_result[f'{models[i].frame_count}_acc1']) / len(testloader)
            epoch_acc5_ = sum(test_result[f'{models[i].frame_count}_acc5']) / len(testloader)

            # model_test_results['running_loss'].append(test_result[f'{models[i].frame_count}_loss'])
            # model_test_results['running_acc1'].append(test_result[f'{models[i].frame_count}_acc1'])
            # model_test_results['running_acc5'].append(test_result[f'{models[i].frame_count}_acc5'])
            # model_test_results['epoch_loss'].append(epoch_loss_)
            # model_test_results['epoch_acc1'].append(epoch_acc1_)
            # model_test_results['epoch_acc5'].append(epoch_acc5_)

            writer.add_scalar(f'testing loss frames={models[i].frame_count}', epoch_loss_, epoch)
            writer.add_scalars('combined testing loss', {f'loss frames={models[i].frame_count}': epoch_loss_}, epoch)
            writer.add_scalars(f'testing accuracy frames={models[i].frame_count}', {'acc1': epoch_acc1_, 'acc5': epoch_acc5_}, epoch)
            writer.add_scalars('combined testing accuracy', {f'acc1 frames={models[i].frame_count}': epoch_acc1_, f'acc5 frames={models[i].frame_count}': epoch_acc5_}, epoch)

            logs['loss'] = epoch_loss
            logs['accuracy'] = epoch_acc1
            logs['accuracy_5'] = epoch_acc5
            logs['val_loss'] = epoch_loss_
            logs['val_accuracy'] = epoch_acc1_
            logs['val_accuracy_5'] = epoch_acc5_

            # liveloss.update(logs)
            # liveloss.send()

            if epoch % 1000 == 0:
                classifier.save_parameters(args.model_params_dir / f'mf-type={args.type}-frames={models[i].frame_count}-batch_size={args.batch_size}-lr={args.learning_rate}_hl={args.hidden_layer_size}_dcp={args.dropout_probability}_epoch={epoch}.pt')

        # training_result.append(model_train_results)
        # testing_result.append(model_test_results)

        classifier.save_parameters(args.model_params_dir / f'mf-type={args.type}-frames={models[i].frame_count}-batch_size={args.batch_size}-lr={args.learning_rate}_hl={args.hidden_layer_size}_dcp={args.dropout_probability}_epoch={args.epoch}.pt')
    
    # return {'training': training_result, 'testing': testing_result}

    # if args.save_params:
    #     classifier.save_parameters(args.save_params)

    # loss = np.concatenate(loss)

    # if args.save_fig:
    #     x = np.linspace(1, len(loss), len(loss), dtype=int)

    #     fig = go.Figure()

    #     fig.add_trace(go.Scatter(
    #         x=x,
    #         y=loss
    #     ))

    #     fig.update_layout(
    #         xaxis_title='batched steps',
    #         yaxis_title='loss',
    #         title='training performance'
    #     )
    #     fig.update_yaxes(type='log')
    #     fig.write_image(args.save_fig)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """
    Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    Args:
        args: CLI Arguments
    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'epic_mf_type={args.type}_epochs={args.epoch}_batch_size={args.batch_size}_lr={args.learning_rate}_hl={args.hidden_layer_size}_dcp={args.dropout_probability}'

    i = 0
    while i < 1000:
        tb_log_dir = Path('datasets/epic-100/runs') / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())

