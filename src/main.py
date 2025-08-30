import torch
import numpy as np
import random
import os
import torchvision.transforms as transforms
from PIL import Image


#data.generate_blurred_data()

folder_path='../Blur_dataset'
#importo tutti i modelli e li metto su gpu/cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CNN=torch.load('models/resnet18.pth')
CNN.eval()
CNN=CNN.to(device)

GRNNResnet=torch.load('models/GRNNResnet.pth')
GRNNResnet=GRNNResnet.to(device)

ViT=torch.load('models/mobileVit.pth')
ViT=ViT.to(device)

GRNNViT=torch.load('models/GRNNVit.pth')
GRNNViT=GRNNViT.to(device)

transform = transforms.Compose([
            transforms.Resize((224, 224)),  # forza ogni immagine a 224x224
            transforms.ToTensor(),
        ])

rnd= random.randint(0, 3400)
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
img=files[rnd]
imgTens=Image.open(folder_path+'/'+img)
imgTens=transform(imgTens)
imgTens=imgTens.unsqueeze(0).to(device)
params=img.split('-')


# test

midStep1=CNN(imgTens)
midStep1= [ out.item() for out in midStep1[0]]
GRNN1Label=[float(params[1]), float(params[2])]
input_tensor = torch.tensor([midStep1] , dtype=torch.float32).to(device)
output1=GRNNResnet(input_tensor)


#qui faccio con il trasformatore
midstep2=ViT(imgTens)
midstep2=[out.item() for out in midstep2[0]]
GRNN2Label=[float(params[1]), float(params[2])]
input_tensor2 = torch.tensor([midstep2], dtype=torch.float32).to(device)
output2=GRNNViT(input_tensor2)


print(f"valori paper: {output1}")
print(f"valori transformatore: {output2}")
print(f"valori veri: {params[0]}, {params[1]}, {params[2]}")


