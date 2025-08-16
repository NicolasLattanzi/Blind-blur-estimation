import torch
from torch.utils.data import *
from torchvision import transforms

import os
import utils
from PIL import Image


class BlurDataset(Dataset):

    def __init__(self, training=True):
        self.root = '../Blur_dataset'
        if training:
            full_path = os.path.join(self.root)
            self.images = [os.path.join(full_path, img) for img in os.listdir(full_path) if img.endswith('.jpg')]
        else: # evaluation
            pass
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        blur_type, blur_parameters = utils.blur_type_from_image_path(img_path)
        blur_type = torch.tensor( blur_type, dtype=str )
        blur_parameters = torch.tensor( blur_parameters, dtype=torch.int32 )

        if self.transform:
            image = self.transform(image)

        return image, blur_type, blur_parameters

def train_test_split(dataset, train=0.5, test=0.5):
    return torch.utils.data.random_split(dataset, [train, test])


############## GENERAZIONE DATA #####################################

src_dir = '../BSDS500/train'
dst_dir = '../Blur_dataset'

# (adesso non fa ancora tutto, lo implemento un po alla volta)
def generate_blurred_data():
    # a partire dal dataset iniziale, estraiamo sezioni di immagini (da 64x64 pixel a
    # 128x128) e applichiamo sopra un blur casuale. Le informazioni di blur sono salvate nel
    # nome dell'immagine.
    # tipi di blur: gaussian, bicubic, motion blur, defocus

    for filename in os.listdir(src_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):

            img = Image.open(os.path.join(src_dir, filename)) #.convert('RGB')
            blur = transforms.GaussianBlur( kernel_size = 11 ) # blur gaussiano
            img_blur = blur(img)

            filename = filename.replace('-', '')
            img_blur.save(os.path.join(dst_dir, '-gaussian-11-', filename))
