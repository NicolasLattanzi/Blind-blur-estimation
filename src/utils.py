# dizionario dei tipi, da usare visto che non esistono tensori stringhe
import os
blur_types = {0: "Gaussian Blur", 1: "Motion Blur", 2: "Defocus Blur"}

# estrazione informazioni blur dal path/nome dell'immagine
def blur_type_from_image_path(path :str):
    filename = os.path.basename(path)
    variables = filename.split('-')
    blur_type = int(variables[0])
    blur_size = int(variables[1])
    blur_param = float(variables[2])

    return [blur_type, blur_size, blur_param]


############ utility ######################


# prende un tensore in input e lo traduce in gaussian, motion o defocus
# (il tensore è pensato per essere l'output del modello di classificazione)
def printable_classif( classif_output ):
    bt = 0
    blur_value = classif_output[0]
    for i in range(1, 3):
        if classif_output[i] > blur_value:
            bt = i
            blur_value = classif_output[i]
    return blur_types[bt], bt

# denormalizza l'output del modello GRNN
def printable_regr( regr_output ):
    p1 = int(regr_output[0] * 9)
    p2 = round(float(regr_output[1] * 360), 3)
    return [p1, p2]

import torch
import torch.nn.functional as F

def to_frequency_domain(images, size=(224, 224)):
    """
    images: [batch, channels, height, width], dtype float
    Output: [batch, 3, size[0], size[1]]
    """
    batch, ch, h, w = images.shape

    # Resize images if needed
    if (h, w) != size:
        images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
    # Convert to grayscale if more than one channel (optional, keeps shape)
    #images_gray = images.mean(dim=1, keepdim=True)
    # Or use all channels independently
    images_freq = []
    for img in images:
        freq_channels = []
        for c in range(img.shape[0]):
            f_img = torch.fft.fft2(img[c])                           # FFT → [h, w] complex
            f_img = torch.fft.fftshift(f_img)                        # Shift DC to center
            magnitude = torch.abs(f_img)                             # Magnitude [h, w] real
            phase = torch.angle(f_img)                               # Phase [h, w] real
            magnitude = torch.log1p(magnitude)                       # Log scale for visibility
            # normalize each feature
            mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            phase_norm = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)
            freq_channels.append(mag_norm)
            freq_channels.append(phase_norm)
        # Stack magnitude and phase for all channels
        freq_stack = torch.stack(freq_channels, dim=0)              # [ch*2, h, w]
        # Use first three channels
        if freq_stack.shape[0] >= 3:
            freq_stack = freq_stack[:3, :, :]
        else:
            # Padding if less than 3 channels
            pad = 3 - freq_stack.shape[0]
            freq_stack = torch.cat([freq_stack, torch.zeros((pad, *freq_stack.shape[1:]), device=freq_stack.device)], dim=0)
        images_freq.append(freq_stack)
    images_freq = torch.stack(images_freq, dim=0)                   # [batch, 3, h, w]
    return images_freq
