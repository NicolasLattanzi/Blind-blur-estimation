import os
from torchvision import transforms
from PIL import Image

src_dir = '../BSDS500/train'
dst_dir = '../Blur_dataset'


def generate_blurred_data():
    # a partire dal dataset iniziale, estraiamo sezioni di immagini (da 64x64 pixel a
    # 128x128) e applichiamo sopra un blur casuale. Le informazioni di blur sono salvate nel
    # nome dell'immagine.
    # tipi di blur: gaussian, bicubic, motion blur, defocus

    for filename in os.listdir(src_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):

            img = Image.open(os.path.join(src_dir, filename)) #.convert('RGB')
            blur = transforms.GaussianBlur( kernel_size=11 ) # blur gaussiano
            img_blur = blur(img)
            img_blur.save(os.path.join(dst_dir, filename))
