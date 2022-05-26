import sys
import yaml
import random
import numpy as np
import torch


def get_tensor_device(x):
    device = x.get_device()
    if device == -1:
        return 'cpu'
    return device


def is_debug_session():
    gettrace = getattr(sys, 'gettrace', None)
    debug_session = not ((gettrace is None) or (not gettrace()))
    return debug_session


def load_config_yml(config_file):
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = config['config']
            return config
    except:
        print('Config file {} is missing'.format(config_file))
        exit(1)


def set_deterministic(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
