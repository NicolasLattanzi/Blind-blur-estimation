
# estrazione informazioni blur dal path/nome dell'immagine
def blur_type_from_image_path(path :str):
    filename = path.split('/')[-1]
    variables = filename.split('-')
    blur_type = variables[0]
    kernel_size = int(variables[1])

    return [blur_type, kernel_size]