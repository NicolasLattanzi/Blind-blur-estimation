import torch
from torch.utils.data import DataLoader

import data
import utils

###### hyper parameters ########

batch_size = 5
num_epochs = 8

##############################

dataset = data.BlurDataset(training=False)
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

eval_size = len(eval_loader)

resnet18 = torch.load('models/resnet18.pth', weights_only=False)
GRNNResnet=torch.load('models/GRNNResnet.pth', weights_only=False)
ViT=torch.load('models/movileVit3.pth', weights_only=False)
GRNNViT=torch.load('models/GRNNVit.pth', weights_only=False)

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
    for epoch in range(num_epochs):
        eval_loss = 0.0
        print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
        for i, (images, blur_types, param1, param2) in enumerate(eval_loader):
            images = images.to(device)
            blur_types = blur_types.to(device)
            param1 = param1.to(device)
            param2 = param2.to(device)
            blur_parameters = torch.tensor([[p1,p2] for p1,p2 in zip(param1, param2)], dtype=torch.float32)

            classif_outputs = resnet18(images) # classification
            final_outputs = GRNNResnet.forward(classif_outputs) # regression
            loss = loss_function(final_outputs, blur_parameters)

            eval_loss += loss.item()

        avg_eval_loss = eval_loss / eval_size
        print(f"Epoch [{epoch+1}/{num_epochs}] evaluation completed. Average Loss: {avg_eval_loss:.4f}")

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
        cout = utils.printable_classif(classif_outputs1[i])
        print(f"blur type: {utils.blur_types[ blur_types[i].item() ]}  /  {cout}")
        b1 = utils.printable_regr(blur_parameters[i])
        b2 = utils.printable_regr(final_outputs1[i])
        print(f"paramaters: {b1}  /  {b2}")

        print('ViT - GRNN')
        cout = utils.printable_classif(classif_outputs2[i])
        print(f"blur type: {utils.blur_types[ blur_types[i].item() ]}  /  {cout}")
        b2 = utils.printable_regr(final_outputs2[i])
        print(f"paramaters: {b1}  /  {b2}")


#single_eval()
complete_eval()
