import torch
from torch.utils.data import *
from torchvision import transforms
from torchvision.transforms import functional

from PIL import Image
from scipy.signal import convolve2d
import os, shutil, cv2, random
import numpy as np
import utils


class BlurDataset(Dataset):

    def __init__(self, training=True):
        if training:
            self.path = '../Blur_dataset'
        else: # evaluation
            self.path = '../Blur_val_dataset'
        self.images = [os.path.join(self.path, img) for img in os.listdir(self.path) if img.endswith('.jpg')]
        
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
        blur_type, blur_param1, blur_param2 = utils.blur_type_from_image_path(img_path)
        blur_type = torch.tensor( blur_type, dtype=torch.int64 )
        blur_param1 = torch.tensor( blur_param1, dtype=torch.int32 )
        blur_param2 = torch.tensor( blur_param2, dtype=torch.int32 )

        if self.transform:
            image = self.transform(image)

        return image, blur_type, blur_param1, blur_param2
    
    # batch di immagini in input
    def augment_data(self, images):
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        outputs = []
        for img_tensor in images:
            img = transforms.ToPILImage()(img_tensor.cpu())
            img = augmentation(img)
            img = transforms.ToTensor()(img)
            outputs.append(img)

        return torch.stack(outputs)


def train_test_split(dataset, train=0.5, test=0.5):
    return torch.utils.data.random_split(dataset, [train, test])



################# GENERAZIONE DATA #####################################


def generate_blurred_data(validation = False):
    # a partire dal dataset iniziale, estraiamo sezioni di immagini (da 64x64 pixel a
    # 128x128) e applichiamo sopra un blur casuale. Le informazioni di blur sono salvate nel
    # nome dell'immagine.
    # tipi di blur: gaussian, motion blur, defocus

    folders = ['../BSDS500/train', '../BSDS500/test', '../DIV2K_train_HR']
    dst_dir = '../Blur_dataset'
    if validation:
        folders = ['../BSDS500/val', '../DIV2K_valid_HR']
        dst_dir = '../Blur_val_dataset'
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # tolgo tutti i file precedenti, così non ammucchiamo vecchi dataset con nuovi
    for filename in os.listdir(dst_dir):
        file_path = os.path.join(dst_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # main loop
    for folder_name in folders:
        for filename in os.listdir(folder_name):
            if not filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue

            img = Image.open(os.path.join(folder_name, filename)).convert('RGB')

            w, h = img.size
            # dimensione del crop
            sizes = [64, 96, 128]
            for size in sizes:
                if w <= size or h <= size: continue

                #scelta randomica della zona dell'immagine per il crop
                rnd_left = random.randint(0, w - size)
                rnd_top = random.randint(0, h - size)
                crop_img = functional.crop(img,rnd_top,rnd_left,size,size)   # top left height width

                # random blurring
                random_blur = int(random.random()*3)
                blur_param_1 = random.randrange(5, 12, 2) # kernel size
                blur_param_2 = 0 # foo
                
                if random_blur == 0:
                    blur_type = 0 # Gaussian Blur
                    blur = transforms.GaussianBlur( kernel_size = blur_param_1 )
                    blurred_img = blur(crop_img)
                elif random_blur == 1:
                    blur_type = 1 # Motion Blur
                    blur_param_1 = random.randrange(16, size)  # blur size
                    blur_param_2 = random.randrange(-360, 360) # motion angle
                    img_array = np.asarray(crop_img)
                    blurred_img = apply_motion_blur(img_array, size = blur_param_1, angle = blur_param_2)
                    blurred_img = Image.fromarray(blurred_img)
                else :
                    blur_type = 2 # Lens (Defocus) Blur
                    blurred_img = apply_lens_blur(crop_img, blur_param_1)
                

                # saving the image
                filename = filename.replace('-', '')
                blurred_img.save( os.path.join(dst_dir, f'{blur_type}-{blur_param_1}-{blur_param_2}-{filename}') )



#################################### BLUR FUNCTIONS ###################################

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
