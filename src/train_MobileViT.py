import tensorflow as tf
from keras import layers as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network
import data

#parametri
batch_size = 16
num_epochs = 8
learning_rate = 0.001

#dataset e separazione in train e test
dataset = data.BlurDataset()
train_dataset, test_dataset = data.train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_size = len(train_loader)
test_size = len(test_loader)

#modello
model=network.MobileViT_XS()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

transform = transforms.Compose([
            transforms.Resize((128, 128)),  # forza ogni immagine a 128x128
            transforms.ToTensor(),
        ])

print('//  starting training  //')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss=0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, blur_types, _, _) in enumerate(train_loader):
        images = dataset.augment_data(images)
        if(images.shape[-1]<128):
            images=transform(images)
        images = images.to(device)
        blur_types = blur_types.to(device)

        output=model(images)
        loss=loss_function(output,blur_types)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {avg_train_loss:.4f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (images, blur_types, _, _) in enumerate(test_loader):
            images = images.to(device)
            blur_types = blur_types.to(device)

            outputs = model(images)
            loss = loss_function(outputs, blur_types)
            test_loss += loss.item()

            #if (i + 1) % 20 == 0:
            #    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_size}], Loss: {loss.item():.4f}")

    avg_test_loss = test_loss / test_size
    print(f"Epoch [{epoch+1}/{num_epochs}] test completed. Average Loss: {avg_test_loss:.4f}\n")

torch.save(model, 'models/movileViT.pth')

