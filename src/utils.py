# dizionario dei tipi, da usare visto che non esistono tensori stringhe
blur_types = {0: "Gaussian Blur", 1: "Motion Blur", 2: "Defocus Blur"}


# estrazione informazioni blur dal path/nome dell'immagine
def blur_type_from_image_path(path :str):
    filename = path.split('/')[-1]
    variables = filename.split('-')
    blur_type = int(variables[0])
    blur_size = int(variables[1])
    blur_param = int(variables[2])

    return [blur_type, blur_size, blur_param]