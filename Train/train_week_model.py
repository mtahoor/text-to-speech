import pandas as pd
import numpy as np
import torch
import random
import albumentations
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import cv2
from tqdm import tqdm  
import cnn_models  # Assuming you have a module called cnn_models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the model for')
args = vars(parser.parse_args())

# Set random seed for reproducibility
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True 
SEED=42
seed_everything(SEED=SEED)

# Check if CUDA is available, otherwise use CPU
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}")

# Load data
df = pd.read_csv('/Users/macbook/Downloads/gestures to speech/Train/data.csv')
X = df.image_path.values
y = df.target.values

# Split data into training and validation sets
(xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, 
                                test_size=0.15, random_state=42))
print(f"Training on {len(xtrain)} images")
print(f"Validation on {len(xtest)} images")

# Define dataset class with data augmentation
class ASLImageDataset(Dataset):
    def __init__(self, path, labels, augmentation=None):
        self.X = path
        self.y = labels
        self.augmentation = augmentation

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        image = cv2.imread(self.X[i])
        if self.augmentation:
            image = self.augmentation(image=image)['image']  # Apply augmentation
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

# Define more aggressive data augmentation
train_augmentation = albumentations.Compose([
    albumentations.Resize(224, 224, always_apply=True),
    albumentations.RandomBrightnessContrast(),
    albumentations.Blur(),
    albumentations.GaussNoise(),
    albumentations.RandomRotate90(),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    albumentations.ShiftScaleRotate(),
])

# Define the model with reduced complexity
class ModifiedCustomCNN(nn.Module):
    def __init__(self):
        super(ModifiedCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 64)
        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Create dataset instances
train_data = ASLImageDataset(xtrain, ytrain, augmentation=train_augmentation)
test_data = ASLImageDataset(xtest, ytest)

# Create data loaders
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize the modified CNN model
model = ModifiedCustomCNN().to(device)
print(model)

# Calculate the total number of parameters and trainable parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Keep learning rate low
criterion = nn.CrossEntropyLoss()

# Function to validate the model
def validate(model, dataloader):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == target).sum().item()
        
        val_loss = running_loss/len(dataloader.dataset)
        val_accuracy = 100. * running_correct/len(dataloader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        
        return val_loss, val_accuracy

# Function to train the model
def fit(model, dataloader):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss/len(dataloader.dataset)
    train_accuracy = 100. * running_correct/len(dataloader.dataset)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
    
    return train_loss, train_accuracy

# Lists to store training and validation loss, accuracy
train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []

# Training loop
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss, train_epoch_accuracy = fit(model, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(model, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()

# Print total time taken for training
print(f"{(end-start)/60:.3f} minutes")

# Plot accuracy
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

# Plot loss
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# Save model
torch.save(model.state_dict(), '/Users/macbook/Downloads/gestures to speech/Train/model.pth')
print('Model saved successfully as model.pth')
