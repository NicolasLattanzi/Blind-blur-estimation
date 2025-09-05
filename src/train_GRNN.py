import torch
from torch.utils.data import DataLoader
import network
import data
import torchvision.transforms as transforms
import utils

#parametri
batch_size = 35 
num_epochs = 13
learning_rate = 0.0007
#dataset e separazione in train e test
dataset = data.BlurDataset(resize=224)
train_dataset, test_dataset = data.train_test_split(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_size = len(train_loader)
test_size = len(test_loader)
resize_transform = transforms.Resize((224, 224))
#modello
model=network.mobilevit_s()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

print('//  starting training  //')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def resize_if_needed(images, size=(224, 224)):
    # images: Tensor [batch, channels, height, width]
    h, w = images.shape[-2], images.shape[-1]
    if (h, w) != size:
        # Resize batch: torchvision Resize expects PIL, but can handle tensors since v0.8
        images = torch.stack([resize_transform(img) for img in images])
    return images

for epoch in range(num_epochs):
    model.train()
    train_loss=0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, blur_types, _, _) in enumerate(train_loader):
        images = dataset.augment_data(images)
        images=resize_if_needed(images)
        images=utils.to_frequency_domain(images)
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
            images=resize_if_needed(images)
            images=utils.to_frequency_domain(images)
            images = images.to(device)
            blur_types = blur_types.to(device)

            outputs = model(images)
            loss = loss_function(outputs, blur_types)
            test_loss += loss.item()

            #if (i + 1) % 20 == 0:
            #    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_size}], Loss: {loss.item():.4f}")

    avg_test_loss = test_loss / test_size
    print(f"Epoch [{epoch+1}/{num_epochs}] test completed. Average Loss: {avg_test_loss:.4f}\n")

torch.save(model, 'models/movileViT5.pth')

