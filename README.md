# Extremely Simple Activation Shaping for Out-of-Distribution Detection

Implementation of **A**ctivation **Sh**aping model for OOD detection. [ICLR2023]

<a href="https://arxiv.org/abs/2209.09858" target="_blank">[Paper]</a> <a href="https://andrijazz.github.io/ash/" target="_blank">[Project Page]</a>

![Activation Shaping method](resources/overview_figure_cropped-min.png)
## Setup

```bash
# create conda env and install dependencies
$ conda env create -f environment.yml
$ conda activate ash
# set environmental variables
$ export DATASETS=<your_path_to_datasets_folder>
$ export MODELS=<your_path_to_checkpoints_folder>
# download datasets and checkpoints
$ bash scripts/download.sh
```
Please download ImageNet dataset manually to `$DATASET` dir by following [this](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) instructions.

## Run
```bash
$ python ood_eval.py --config config/imagenet_config.yml --use-gpu --use-tqdm
```

## Usage

```python
import torch
import torch.nn as nn

from ash import ash_b, get_score

class Net(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        
        # add activation shaping to forward pass of your network
        x = ash_b(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = Net()
for i, data in enumerate(testloader):
    inputs, labels = data
    logits = net(inputs)
    
    # get ood predictions
    ood_prediction = get_score(logits)
```
## TODO

- [ ] Test our OOD approach to a benchmark https://github.com/IML-DKFZ/fd-shifts (Paul reached out to us during our poster presentation)
- [x] Combine with the other OOD approach https://github.com/deeplearning-wisc/cider, which seems to be complementary (Ousmane reached out to us durin our poster presentation)

## References

```bibtex
@inproceedings{sun2021react,
  title={ReAct: Out-of-distribution Detection With Rectified Activations},
  author={Sun, Yiyou and Guo, Chuan and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

```bibtex
@inproceedings{sun2022dice,
  title={DICE: Leveraging Sparsification for Out-of-Distribution Detection},
  author={Sun, Yiyou and Li, Yixuan},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
      
## Citations

If you use our codebase, please cite our work:

```bibtex
@article{djurisic2022ash,
    url = {https://arxiv.org/abs/2209.09858},
    author = {Djurisic, Andrija and Bozanic, Nebojsa and Ashok, Arjun and Liu, Rosanne},
    title = {Extremely Simple Activation Shaping for Out-of-Distribution Detection},
    publisher = {arXiv},
    year = {2022},
    }
```
