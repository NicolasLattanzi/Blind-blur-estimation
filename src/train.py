import torch
from torch.utils.data import DataLoader

import network
import data


dataset = data.BlurDataset("..\CCPD2019")
model = network.build_resnet()