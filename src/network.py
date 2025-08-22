import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet():
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    OUTPUTS = 3
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model


class GRNN(nn.Module):
    def __init__(self, train_data, train_labels, spread=1.0):
        self.spread = spread
        self.train_inputs = train_data    # tensore (N, 3)
        self.train_outputs = train_labels # tensore (N, 2)

    def forward(self, x):
        # Calcola distanze al quadrato tra x e tutti i dati di training
        # x shape: (batch_size, 3)
        # train_inputs shape: (N, 3)
        # risultato: (batch_size, N)
        diff = x.unsqueeze(1) - self.train_inputs.unsqueeze(0) #forma: batch_size, N, 3
        dist_sq = torch.sum(diff**2, dim=2)

        # Calcola pesi (con formula gaussiana)
        weights = torch.exp(-dist_sq / (2 * self.spread**2))

        # Calcola somma pesata degli output dei training
        weighted_outputs = torch.matmul(weights, self.train_outputs)  # shape (batch_size, 2)
        weights_sum = weights.sum(dim=1, keepdim=True)                 # shape (batch_size, 1)

        # media pesata
        output = weighted_outputs / weights_sum
        return output

