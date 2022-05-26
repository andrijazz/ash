# Reference
# https://github.com/ShiyuLiang/odin-pytorch/blob/34e53f5a982811a0d74baba049538d34efc0732d/code/calData.py#L208-L212
# https://github.com/ShiyuLiang/odin-pytorch/blob/34e53f5a982811a0d74baba049538d34efc0732d/code/calData.py#L333-L336
# https://github.com/kobybibas/pnml_ood_detection/blob/ad0e29c1f35347d9437fe453fac59824c8a60c8d/src/dataset_utils.py#L70
# https://github.com/kobybibas/pnml_ood_detection/blob/ad0e29c1f35347d9437fe453fac59824c8a60c8d/src/dataset_utils.py#L108

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RandomData(Dataset):
    def __init__(self, num_samples, is_gaussian=True, transform=None):
        self.num_samples = num_samples
        self.gaussian = is_gaussian
        self.transform = transform
        self.targets = [-1] * self.num_samples
        if self.gaussian:
            self.data = 255 * np.random.randn(self.num_samples, 32, 32, 3) + 255 / 2
            self.data = np.clip(self.data, 0, 255).astype("uint8")
        else:
            self.data = np.random.randint(0, 255, (self.num_samples, 32, 32, 3)).astype("uint8")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
