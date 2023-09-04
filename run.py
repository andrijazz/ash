#!/usr/bin/env python

import argparse
import os
from functools import partial

import ray
from ray import tune

from ood_eval import ood_eval
from utils.utils import is_debug_session


def cifar100_vs_cifar10():
    config = {
        "id_dataset": "cifar100",
        "ood_datasets": ["cifar10"],
        "model_name": "densenet100",
        "train_restore_file": "densenet100_cifar100.pth",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["ash_s@65", "ash_s@80", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@80", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@80", "ash_p@90", "ash_p@95", "ash_p@99", "energy"])
    }
    run(config)


def cifar10_vs_cifar100():
    config = {
        "id_dataset": "cifar10",
        "ood_datasets": ["cifar100"],
        "model_name": "densenet100",
        "train_restore_file": "densenet100_cifar10.pth",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["ash_s@65", "ash_s@80", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@80", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@80", "ash_p@90", "ash_p@95", "ash_p@99", "energy"])
    }
    run(config)


def vit():
    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "vit",
        "train_restore_file": "",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["energy", "ash_s@65", "ash_s@90", "ash_s@95", "ash_s@99", "ash_b@65", "ash_b@90", "ash_b@95", "ash_b@99"])
    }
    run(config)


def densenet_imagenet():
    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "densenet_imagenet",
        "train_restore_file": "",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["energy", "ash_s@65", "ash_s@90", "ash_s@95", "ash_s@99", "ash_b@65", "ash_b@90", "ash_b@95", "ash_b@99"])
    }
    run(config)


def convnext():
    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "convnext",
        "train_restore_file": "",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["energy", "ash_s@65", "ash_s@90", "ash_s@95", "ash_s@99", "ash_b@65", "ash_b@90", "ash_b@95", "ash_b@99"])
    }
    run(config)


def vgg():
    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "vgg16",
        "train_restore_file": "vgg16-397923af.pth",
        "batch_size": 200,
        "scoring_method": "energy",
        "method": tune.grid_search(["ash_s@65", "ash_s@95", "ash_b@65", "ash_b@90", "ash_b@95"])
    }
    run(config)


def imagenet():
    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "resnet50",
        "train_restore_file": "resnet50-19c8e357.pth",
        "batch_size": 200,
        "scoring_method": tune.grid_search(["energy", "msp"]),
        "method": tune.grid_search(["ash_s@65", "ash_s@70", "ash_s@75", "ash_s@80", "ash_s@85", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@70", "ash_b@75", "ash_b@80", "ash_b@85", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@70", "ash_p@75", "ash_p@80", "ash_p@85", "ash_p@90", "ash_p@95", "ash_p@99"])
    }
    run(config)

    config = {
        "id_dataset": "imagenet",
        "ood_datasets": ["inaturalist", "sun", "places", "textures"],
        "model_name": "mobilenetv2",
        "train_restore_file": "mobilenet_v2-b0353104.pth",
        "batch_size": 200,
        "scoring_method": tune.grid_search(["energy", "msp"]),
        "method": tune.grid_search(["ash_s@65", "ash_s@70", "ash_s@75", "ash_s@80", "ash_s@85", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@70", "ash_b@75", "ash_b@80", "ash_b@85", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@70", "ash_p@75", "ash_p@80", "ash_p@85", "ash_p@90", "ash_p@95", "ash_p@99"])
    }
    run(config)


def cifar():
    config = {
        "id_dataset": "cifar10",
        "ood_datasets": ["svhn", "lsuncrop", "lsunresize", "isun", "textures", "places365"],
        "model_name": "densenet100",
        "train_restore_file": "densenet100_cifar10.pth",
        "batch_size": 200,
        "scoring_method": tune.grid_search(["energy", "msp"]),
        # "method": tune.grid_search(["ash_s@65", "ash_s@70"])
        "method": tune.grid_search(["ash_s@65", "ash_s@70", "ash_s@75", "ash_s@80", "ash_s@85", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@70", "ash_b@75", "ash_b@80", "ash_b@85", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@70", "ash_p@75", "ash_p@80", "ash_p@85", "ash_p@90", "ash_p@95", "ash_p@99"])
    }
    run(config)

    config = {
        "id_dataset": "cifar100",
        "ood_datasets": ["svhn", "lsuncrop", "lsunresize", "isun", "textures", "places365"],
        "model_name": "densenet100",
        "train_restore_file": "densenet100_cifar100.pth",
        "batch_size": 200,
        "scoring_method": tune.grid_search(["energy", "msp"]),
        "method": tune.grid_search(["ash_s@65", "ash_s@70", "ash_s@75", "ash_s@80", "ash_s@85", "ash_s@90", "ash_s@95", "ash_s@99",
                                    "ash_b@65", "ash_b@70", "ash_b@75", "ash_b@80", "ash_b@85", "ash_b@90", "ash_b@95", "ash_b@99",
                                    "ash_p@65", "ash_p@70", "ash_p@75", "ash_p@80", "ash_p@85", "ash_p@90", "ash_p@95", "ash_p@99"])
    }
    run(config)


def run(config):
    gpus_per_trial = 1
    analysis = tune.run(partial(ood_eval, use_gpu=True, use_tqdm=True),
                        config=config,
                        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
                        log_to_file=True)
    print(analysis)


def exec(args):
    function_name = args.job
    eval(function_name)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, required=True, help="Specify job type [cifar / imagenet]")
    parser.add_argument("--address", type=str, help="Ray address to use to connect to a cluster.")
    args = parser.parse_args()
    ray.init(address=args.address, local_mode=is_debug_session())
    exec(args)
