import torch
from torch.utils.data import DataLoader

import data

###### hyper parameters ########

batch_size = 16
num_epochs = 8

##############################

dataset = data.BlurDataset()
train_dataset, test_dataset = data.train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_size = len(train_loader)
test_size = len(test_loader)

resnet18 = torch.load('models/resnet18.pth')
GRNN = torch.load('models/GRNN.pth')
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet18 = resnet18.to(device)

print('//  starting resnet + GRNN testing  //')

loss_function = torch.nn.MSELoss()
resnet18.eval()

for epoch in range(num_epochs):
    train_loss = 0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, blur_types, param1, param2) in enumerate(test_loader):
        images = images.to(device)
        blur_types = blur_types.to(device)
        param1 = param1.to(device)
        param2 = param2.to(device)

        classif_outputs = resnet18(images) # classification
        final_outputs = GRNN.forward(classif_outputs) # regression
        print(final_outputs)
        break
        loss = loss_function(final_outputs, blur_types)

        train_loss += loss.item()

        # printing error every X batch
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_size}], Loss: {loss.item():.4f}")
        #if (i == 2): break

    avg_train_loss = train_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {avg_train_loss:.4f}")