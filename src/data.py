import torch
from torch.utils.data import *
from torchvision import transforms
from torchvision.transforms import functional

from PIL import Image
from scipy.signal import convolve2d
import os
import utils
import cv2
import numpy as np
import random


class BlurDataset(Dataset):

    def __init__(self, path, training=True):
        self.root = path
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
        blur_type = torch.tensor( blur_type, dtype=torch.int64 )
        blur_parameters = torch.tensor( blur_parameters, dtype=torch.int32 )

        if self.transform:
            image = self.transform(image)

        return image, blur_type, blur_parameters

def train_test_split(dataset, train=0.5, test=0.5): # aggiungere validate?
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

            img = Image.open(os.path.join(src_dir, filename)).convert('RGB')

            # cropping
            w, h = img.size
            #scelta randomica della dimensione del crop
            rnd=random.randint(0,3)
            if rnd==0:  a=64
            if rnd==1:  a=96
            else:       a=128

            if w <= a or h <= a: continue
            #scelta randomica della zona dell'immagine per il crop
            rnd_left=random.randint(0, w - a)
            rnd_top=random.randint(0, h - a)
            img=functional.crop(img,rnd_top,rnd_left,a,a)   # top left height width

            # random blurring
            random_blur = int(random.random()*3)
            kernel_size = random.randrange(5, 12, 2)
            if random_blur ==0:
                blur_type = 0 # Gaussian Blur
                blur = transforms.GaussianBlur( kernel_size = kernel_size )
                blurred_img = blur(img)
            if random_blur ==1:
                blur_type == 1 # Motion Blur
                img_array = np.asarray(img)
                blurred_img = apply_motion_blur(img_array, kernel_size, 90)
                blurred_img = Image.fromarray(blurred_img)
            else :
                #qui aggiungere il terzo blur che non ho capito se è pronto o no 
                b=10 #giusto per non darmi errore nell'else
            

            # saving the image
            filename = filename.replace('-', '')
            blurred_img.save( os.path.join(dst_dir, f'{blur_type}-{kernel_size}-{filename}') )


#size - in pixels, size of motion blur
#angle - in degrees, direction of motion blur
def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 

#radius - size of the disc kernel
def apply_lens_blur(img, radius):
    # Gamma correction: img^3
    arr = np.array(img).astype(np.float32) / 255
    gamma_img = np.power(arr, 3)
    gamma_img = (gamma_img * 255).astype(np.uint8)
    gamma_img = Image.fromarray(gamma_img)
    
    # Disc blur
    bokeh = convolve_disc(gamma_img, radius)
    bokeh_arr = np.array(bokeh).astype(np.float32) / 255
    bokeh_arr = np.cbrt(bokeh_arr)
    bokeh = Image.fromarray((bokeh_arr * 255).astype(np.uint8))
    
    blur_img = convolve_disc(img, radius)
    
    final = np.maximum(np.array(bokeh), np.array(blur_img)).astype(np.uint8)
    return Image.fromarray(final)

def disc_kernel(radius):
    # Crea un kernel a forma di disco
    size = 2*radius+1
    Y, X = np.ogrid[:size, :size]
    dist = (X - radius)**2 + (Y - radius)**2
    mask = dist <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1
    kernel /= np.sum(kernel)
    return kernel

def convolve_disc(image, radius):
    arr = np.array(image)
    kernel = disc_kernel(radius)
    if arr.ndim == 3:  # Color image
        channels = [convolve2d(arr[:,:,i], kernel, mode='same', boundary='symm') for i in range(3)]
        arr_conv = np.stack(channels, axis=-1)
    else:  # Grayscale
        arr_conv = convolve2d(arr, kernel, mode='same', boundary='symm')
    arr_conv = np.clip(arr_conv, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_conv)

