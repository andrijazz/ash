# ASH

Implementation of *Ash* model, from the following paper:

[Extremely Simple Activation Shaping for Out-of-Distribution Detection](). NeurIPS 2022.\
Andrija Djurisic, Arjun Ashok, Nebojsa Bozanic and Rosanne Liu\
ML Collective, Google Brain

## Setup

```bash
# install dependencies
$ pip install torch torchvision numpy

# download datasets
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
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    logits = net(inputs)
    
    # get ood predictions
    ood_prediction = get_score(logits)
```
