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
    return blur_types[bt]

# denormalizza l'output del modello GRNN
def printable_regr( regr_output ):
    p1 = int(regr_output[0] * 128)
    p2 = int(regr_output[1] * 360)
    return [p1, p2]