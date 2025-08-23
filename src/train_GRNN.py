import torch
from torch.utils.data import DataLoader

import network
import data

###### hyper parameters ########

batch_size = 1

###############################

dataset = data.BlurDataset()
train_dataset, test_dataset = data.train_test_split(dataset, 1, 0)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

data_size = len(data_loader)

resnet18 = torch.load('models/resnet18.pth')
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet18 = resnet18.to(device)

########################### preparing GRNN data ###########################

GRNN_TRAIN_DATA = []
GRNN_LABELS = []

resnet18.eval()
# con la resnet18 caricata, scorro tutte le immagini di train e registro gli output in GRNN_TRAIN_DATA e GRNN_LABELS
# le liste verranno usate per il mega hidden layer del modello
for i, (images, _, param1, param2) in enumerate(data_loader):
    images = images.to(device)
    param1 = param1.to(device)
    param2 = param2.to(device)

    outputs = resnet18(images)
    outputs = [ out.item() for out in outputs[0] ]
    GRNN_TRAIN_DATA.append(outputs)
    GRNN_LABELS.append([param1[0].item(), param2[0].item()])

GRNN = network.GRNN(GRNN_TRAIN_DATA, GRNN_LABELS)
torch.save(GRNN, "models/GRNN.pth")

