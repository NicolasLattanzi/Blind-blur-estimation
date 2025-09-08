import torch
from torch.utils.data import DataLoader

import data
import utils

###### hyper parameters ########
#per resnet commentare riga 56!

batch_size = 5

##############################

dataset = data.BlurDataset(training=False)
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

eval_size = len(eval_loader)

resnet18 = torch.load('models/resnet18.pth', map_location=torch.device('cpu'))
GRNNResnet=torch.load('models/GRNNResnet.pth')
ViT=torch.load('models/movileViTDef.pth', map_location=torch.device('cpu'))
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
    total_dict = {"Gaussian Blur" : 0, "Motion Blur" : 0, "Defocus Blur" : 0}
    false_dict = {"Gaussian Blur" : {"Motion Blur" : 0, "Defocus Blur" : 0}, 
                  "Motion Blur" : {"Gaussian Blur" : 0, "Defocus Blur" : 0}, 
                  "Defocus Blur" : {"Gaussian Blur" : 0, "Motion Blur" : 0}}
    kernel_err_dict = {"Gaussian Blur" : 0, "Motion Blur" : 0, "Defocus Blur" : 0}

    with torch.no_grad():
        for i, (images, blur_types, param1, param2) in enumerate(eval_loader):
            images = images.to(device)
            blur_types = blur_types.to(device)
            param1 = param1.to(device)
            param2 = param2.to(device)
            blur_parameters = torch.tensor([[p1, p2] for p1,p2 in zip(param1, param2)], dtype=torch.float32)


            images=utils.to_frequency_domain(images)   #commentare questa riga per resnet!
            classif_outputs = ViT(images) # classification
            final_outputs = GRNNViT.forward(classif_outputs) # regression

            kernel_outputs = torch.tensor([x.item() for x, _ in final_outputs])
            regression_outputs = torch.tensor([x.item() for _, x in final_outputs])
            eval_loss += loss_function(final_outputs, blur_parameters)
            loss1 = loss_function(kernel_outputs, param1)
            loss2 = loss_function(regression_outputs, param2)

            # un po intricato, ma funziona
            for j in range(len(classif_outputs)):
                predicted_blur, predicted_out = utils.printable_classif(classif_outputs[j]) # blur name - number
                if blur_types[j] == predicted_out:
                    true_positives += 1
                    true_dict[ predicted_blur ] += 1
                    kernel_err_dict[predicted_blur] += loss1
                    if predicted_out == 0: # gaussian
                        gaussian_loss += loss2
                        gaussian_counter += 1
                    elif predicted_out == 1: # motion
                        motion_loss += loss2
                        motion_counter += 1
                else:
                    # registro gli sbagli qui
                    false_positives += 1
                    true_blur = utils.blur_types[blur_types[j].item()]
                    false_dict[true_blur][predicted_blur] += 1
                total_dict[ predicted_blur ] += 1
            
            torch.cuda.empty_cache()

    total = true_positives + false_positives
    print(f"evaluation completed, average loss: {eval_loss/eval_size}")

    print('positives: ', true_positives, ' / ', total)
    print(true_dict)
    print('total blur types: \n', total_dict)
    print('false_dict: \n', false_dict)

    print('GRNN parameter error:')
    print(f'average gaussian loss: {gaussian_loss / gaussian_counter}')
    print(f'average motion loss: {motion_loss / motion_counter}')
    print('GRNN kernel error:')
    print(f'avg gaussian kernel loss: {kernel_err_dict["Gaussian Blur"] / total_dict["Gaussian Blur"]}')
    print(f'avg Motion kernel loss: {kernel_err_dict["Motion Blur"] / total_dict["Motion Blur"]}')
    print(f'avg Defocus kernel loss: {kernel_err_dict["Defocus Blur"] / total_dict["Defocus Blur"]}')

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

'''

'''