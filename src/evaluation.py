import torch
from torch.utils.data import DataLoader

import data
import utils

###### hyper parameters ########

batch_size = 10
num_epochs = 8

##############################

dataset = data.BlurDataset(training=False)
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

eval_size = len(eval_loader)

resnet18 = torch.load('models/resnet18.pth')
GRNN = torch.load('models/GRNN.pth')
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet18 = resnet18.to(device)
GRNN = GRNN.to(device)

print('//  starting resnet + GRNN testing  //')

loss_function = torch.nn.MSELoss()
resnet18.eval()
GRNN.eval()

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
            final_outputs = GRNN.forward(classif_outputs) # regression
            loss = loss_function(final_outputs, blur_parameters)

            eval_loss += loss.item()

        avg_eval_loss = eval_loss / eval_size
        print(f"Epoch [{epoch+1}/{num_epochs}] evaluation completed. Average Loss: {avg_eval_loss:.4f}")

def single_eval():
    eval_loss = 0.0
    for i, (images, blur_types, param1, param2) in enumerate(eval_loader):
        images = images.to(device)
        blur_types = blur_types.to(device)
        param1 = param1.to(device)
        param2 = param2.to(device)
        blur_parameters = torch.tensor([[p1,p2] for p1,p2 in zip(param1, param2)], dtype=torch.float32)

        classif_outputs = resnet18(images) # classification
        final_outputs = GRNN.forward(classif_outputs) # regression
        loss = loss_function(final_outputs, blur_parameters)

        eval_loss += loss.item()
        break

    for i in range(batch_size):
        print(i)
        cout = utils.printable_classif(classif_outputs[i])
        print(f"blur type: {utils.blur_types[ blur_types[i].item() ]}  /  {cout}")
        b1 = utils.printable_regr(blur_parameters[i])
        b2 = utils.printable_regr(final_outputs[i])
        print(f"paramaters: {b1}  /  {b2}")

    print(f"evaluation loss: {eval_loss:.4f}")


single_eval()
#complete_eval()