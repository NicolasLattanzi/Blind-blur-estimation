import torch
from torch.utils.data import DataLoader

import network
import data

###### hyper parameters ########

batch_size = 16
num_epochs = 8
learning_rate = 0.001

###############################

dataset = data.BlurDataset()
train_dataset, test_dataset = data.train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_size = len(train_loader)
test_size = len(test_loader)

model = network.build_resnet()
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

print('//  starting training  //')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, blur_types, _, _) in enumerate(train_loader):
        images = images.to(device)
        blur_types = blur_types.to(device)

        # forward step
        outputs = model(images)
        loss = loss_function(outputs, blur_types)

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # printing error every X batch
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_size}], Loss: {loss.item():.4f}")
        #if (i == 2): break

    avg_train_loss = train_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {avg_train_loss:.4f}")

torch.save(model, 'models/resnet18.pth')