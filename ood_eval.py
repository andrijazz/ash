#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ash import get_score
from datasets.dataset_factory import build_dataset, get_num_classes
from models.model_factory import build_model
from utils.metrics import compute_in, compute_traditional_ood
from utils.utils import is_debug_session, load_config_yml, set_deterministic


def eval_id_dataset(model, transform, dataset_name, output_dir, batch_size, use_gpu, use_tqdm):
    print(f'Processing {dataset_name} dataset.')
    dataset = build_dataset(dataset_name, transform, train=False)

    # setup dataset
    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 2, 'pin_memory': True, 'generator': g, 'worker_init_fn': seed_worker}

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    with torch.no_grad():

        if use_tqdm:
            progress_bar = tqdm(total=len(dataloader))

        f1 = open(os.path.join(output_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(output_dir, "in_labels.txt"), 'w')
        for i, samples in enumerate(dataloader):
            images = samples[0]
            labels = samples[1]

            # Create non_blocking tensors for distributed training
            if use_gpu:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            outputs = outputs.detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)
            confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            scores = get_score(logits, config['scoring_method'])
            for score in scores:
                f1.write("{}\n".format(score))

            if use_tqdm:
                progress_bar.update()

        f1.close()
        g1.close()

        if use_tqdm:
            progress_bar.close()


def eval_ood_dataset(model, transform, dataset_name, output_dir, batch_size, use_gpu, use_tqdm):
    print(f'Processing {dataset_name} dataset.')
    dataset = build_dataset(dataset_name, transform, train=False)

    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 2, 'pin_memory': True, 'generator': g, 'worker_init_fn': seed_worker}

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    with torch.no_grad():

        if use_tqdm:
            progress_bar = tqdm(total=len(dataloader))

        f1 = open(os.path.join(output_dir, f"{dataset_name}.txt"), 'w')
        for i, samples in enumerate(dataloader):
            images = samples[0]

            # Create non_blocking tensors for distributed training
            if use_gpu:
                images = images.cuda(non_blocking=True)

            logits = model(images)
            scores = get_score(logits, config['scoring_method'])
            for score in scores:
                f1.write("{}\n".format(score))

            if use_tqdm:
                progress_bar.update()

        f1.close()

        if use_tqdm:
            progress_bar.close()


def ood_eval(config, use_gpu, use_tqdm):
    num_classes = get_num_classes(config['id_dataset'])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # construct the model
    model, transform = build_model(config['model_name'], num_classes=num_classes)
    checkpoint = os.path.join(os.getenv('MODELS'), config['train_restore_file'])
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    if use_gpu:
        model = model.cuda()

    eval_id_dataset(model, transform, config['id_dataset'], output_dir, config['batch_size'], use_gpu, use_tqdm)
    for ood_dataset in config['ood_datasets']:
        eval_ood_dataset(model, transform, ood_dataset, output_dir, config['batch_size'], use_gpu, use_tqdm)

    print(f"{config['method']} - {config['id_dataset']} stats")
    compute_traditional_ood(output_dir, config['ood_datasets'], config['scoring_method'])
    compute_in(output_dir, config['scoring_method'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU")
    parser.add_argument("--use-tqdm", action="store_true", default=False, help="Enables progress bar")
    args = parser.parse_args()
    config = load_config_yml(args.config)
    os.environ['ash_method'] = config['method']
    g, seed_worker = set_deterministic()
    ood_eval(config, args.use_gpu, args.use_tqdm)
