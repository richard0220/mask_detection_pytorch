import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import models, transforms
from datetime import datetime

# Mask custom dataset
class MaskDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.mask_dir = os.path.join(dir, 'mask')
        self.no_mask_dir = os.path.join(dir, 'no_mask')
        mask_images = os.listdir(self.mask_dir)
        no_mask_images = os.listdir(self.no_mask_dir)
        self.datas = mask_images + no_mask_images
        self.labels = [1 for i in range(len(mask_images))] + [0 for i in range(len(no_mask_images))]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        if self.labels[idx] == 1:
            image = Image.open(os.path.join(self.mask_dir, self.datas[idx])).convert('RGB')
            image = self.transform(image)
        else:
            image = Image.open(os.path.join(self.no_mask_dir, self.datas[idx])).convert('RGB')
            image = self.transform(image)

        return image, self.labels[idx]


# Load pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Modify the final layer for binary classification
model.classifier[1] = nn.Linear(model.last_channel, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_dir = "./data/train"
dataset = MaskDataset(train_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Loss Function
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# Training
epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}')

# evaluation
model.eval()
correct = 0
total = 0
threshold = 0.5

test_dir = "./data/test"
dataset = MaskDataset(test_dir, transform)
dataloader = dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        pred = (torch.sigmoid(outputs) > threshold).float()
        total += labels.size(0)
        correct += (pred == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model
time = datetime.now()
torch.save(model.state_dict(), 'mask_detect' + str(time.month) + '_' + str(time.day) + '_' + str(time.hour) + '_' + str(time.minute) + '.pth')