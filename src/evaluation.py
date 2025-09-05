import torch
from torch.utils.data import DataLoader

import data
import utils

###### hyper parameters ########\
#per resnet commentare riga 56!

batch_size = 5

##############################

dataset = data.BlurDataset(training=False)
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

eval_size = len(eval_loader)

resnet18 = torch.load('models/resnet18.pth', map_location=torch.device('cpu'))
GRNNResnet=torch.load('models/GRNNResnet.pth')
ViT=torch.load('models/movileVit4.pth', map_location=torch.device('cpu'))
GRNNViT=torch.load('models/GRNNVit.pth')

# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet18 = resnet18.to(device)
ViT=ViT.to(device)
GRNNResnet=GRNNResnet.to(device)
GRNNViT=GRNNViT.to(device)

print('//  starting resnet + GRNN testing  //')

loss_function = torch.nn.MSELoss()
resnet18.eval()
ViT.eval()
GRNNResnet.eval()
GRNNViT.eval()

def complete_eval():
    eval_loss = 0
    gaussian_loss = gaussian_counter = 0
    motion_loss = motion_counter = 0
    true_positives = false_positives = 0
    true_dict = {"Gaussian Blur" : 0, "Motion Blur" : 0, "Defocus Blur" : 0}
    false_dict = {"Gaussian Blur" : 0, "Motion Blur" : 0, "Defocus Blur" : 0}

    with torch.no_grad():
        for i, (images, blur_types, param1, param2) in enumerate(eval_loader):
            images = images.to(device)
            blur_types = blur_types.to(device)
            param1 = param1.to(device)
            param2 = param2.to(device)
            blur_parameters = torch.tensor([[p1,p2] for p1,p2 in zip(param1, param2)], dtype=torch.float32)


            images=utils.to_frequency_domain(images)   #commentare questa riga per resnet!
            classif_outputs = ViT(images) # classification
            final_outputs = GRNNViT.forward(classif_outputs) # regression
            loss = loss_function(final_outputs, blur_parameters)
            eval_loss += loss.item()

            for j in range(len(classif_outputs)):
                predicted_blur, predicted_out = utils.printable_classif(classif_outputs[j]) # blur name - number
                if blur_types[j] == predicted_out:
                    true_positives += 1
                    true_dict[ predicted_blur ] += 1
                    if predicted_out == 0: # gaussian
                        gaussian_loss += loss
                        gaussian_counter += 1
                    elif predicted_out == 1: # motion
                        motion_loss += loss
                        motion_counter += 1
                else:
                    false_positives += 1
                    false_dict[ predicted_blur ] += 1
            
            torch.cuda.empty_cache()

    avg_eval_loss = eval_loss / eval_size
    print(f"evaluation completed. Average Loss: {avg_eval_loss:.4f}")

    total = true_positives + false_positives
    print('positives: ', true_positives, ' / ', total)
    print(true_dict)
    print('error dict: ')
    print(false_dict)

    print('GRNN error:')
    print(f'average gaussian loss: {gaussian_loss / gaussian_counter}')
    print(f'average motion loss: {motion_loss / motion_counter}')

def single_eval():
    for i, (images, blur_types, param1, param2) in enumerate(eval_loader):
        images = images.to(device)
        blur_types = blur_types.to(device)
        param1 = param1.to(device)
        param2 = param2.to(device)
        blur_parameters = torch.tensor([[p1,p2] for p1,p2 in zip(param1, param2)], dtype=torch.float32)

        classif_outputs1 = resnet18(images) # classification
        classif_outputs2 = ViT(images) # classification
        final_outputs1 = GRNNResnet.forward(classif_outputs1) # regression
        final_outputs2 = GRNNResnet.forward(classif_outputs2) # regression
        break

    for i in range(batch_size):
        print(i)
        
        print('resnet - GRNN')
        cout, _ = utils.printable_classif(classif_outputs1[i])
        print(f"blur type: {utils.blur_types[ blur_types[i].item() ]}  /  {cout}")
        b1 = utils.printable_regr(blur_parameters[i])
        b2 = utils.printable_regr(final_outputs1[i])
        print(f"paramaters: {b1}  /  {b2}")

        print('ViT - GRNN')
        cout, _ = utils.printable_classif(classif_outputs2[i])
        print(f"blur type: {utils.blur_types[ blur_types[i].item() ]}  /  {cout}")
        b2 = utils.printable_regr(final_outputs2[i])
        print(f"paramaters: {b1}  /  {b2}")


#single_eval()
complete_eval()
