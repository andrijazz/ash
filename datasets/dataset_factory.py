import os
import torch
import torchvision
import torchvision.datasets

from datasets.random_dataset import RandomData
from utils.svhn_loader import SVHN


def build_dataset(dataset_name, transform, train=False):
    dataset_dir = os.getenv('DATASETS')

    # cifar10
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(dataset_dir, transform=transform, train=train, download=False)
        return dataset

    if dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(dataset_dir, transform=transform, train=train, download=False)
        return dataset

    if dataset_name == "svhn":
        split = 'train' if train else 'test'
        # dataset = torchvision.datasets.SVHN(os.path.join(dataset_dir, "SVHN"),
        #                                     transform=transform,
        #                                     split=split,
        #                                     download=False)
        dataset = SVHN(os.path.join(dataset_dir, "SVHN"), transform=transform, split='test', download=False)
        return dataset

    # gaussian
    if dataset_name == 'gaussian':
        dataset = RandomData(num_samples=10000, is_gaussian=True, transform=transform)
        return dataset

    if dataset_name == 'uniform':
        dataset = RandomData(num_samples=10000, is_gaussian=False, transform=transform)
        return dataset

    if dataset_name == 'lsuncrop':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'LSUN'), transform=transform)
        return dataset

    if dataset_name == 'lsunresize':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'LSUN_resize'), transform=transform)
        return dataset

    if dataset_name == 'tinyimagenetcrop':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'Imagenet'), transform=transform)
        return dataset

    if dataset_name == 'tinyimagenetresize':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'Imagenet_resize'), transform=transform)
        return dataset

    if dataset_name == 'isun':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'iSUN'), transform=transform)
        return dataset

    # imagenet30
    if dataset_name == "imagenet30":
        mode_path = "one_class_train" if train else "one_class_test"
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'imagenet30', mode_path), transform=transform)
        return dataset

    # imagenet-A
    if dataset_name == "imageneta":
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'imagenet-a'), transform=transform)
        return dataset

    # imagenet-R
    if dataset_name == "imagenetr":
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'imagenet-r'), transform=transform)
        return dataset

    # imagenet
    if dataset_name == "imagenet":
        mode_path = "train" if train else "val"
        external_disk = os.getenv('EXTERNAL_DRIVE') if os.getenv('EXTERNAL_DRIVE') else dataset_dir
        dataset = torchvision.datasets.ImageFolder(os.path.join(external_disk, 'imagenet', mode_path), transform=transform)
        return dataset

    if dataset_name == 'inaturalist':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'iNaturalist'), transform=transform)
        return dataset

    if dataset_name == 'sun':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'SUN'), transform=transform)
        return dataset

    if dataset_name == 'places':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'Places'), transform=transform)
        return dataset

    if dataset_name == 'places365':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'places365'), transform=transform)
        dataset_subset = torch.utils.data.Subset(dataset, list(range(10000)))
        return dataset_subset

    if dataset_name == 'textures':
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'dtd', 'images'), transform=transform)
        return dataset

    if dataset_name == "imagenette":
        mode_path = "train" if train else "val"
        dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'imagenette2-320', mode_path), transform=transform)
        return dataset
    exit(f'{dataset_name} dataset is not supported')


def get_num_classes(in_dataset_name):
    if in_dataset_name == 'cifar10':
        return 10
    if in_dataset_name == 'cifar100':
        return 100
    if in_dataset_name == 'svhn':
        return 10
    if in_dataset_name == 'imagenet':
        return 1000
    if in_dataset_name == 'imagenet30':
        return 30
    if in_dataset_name == 'imagenette':
        return 10
    exit(f'Unsupported in-dist dataset: f{in_dataset_name}')

