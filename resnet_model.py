import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Custom dataset to load frames from videos
class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_dirs = [os.path.join(data_dir, 'real'), os.path.join(data_dir, 'fake')]
        self.frames = []

        # Iterate through real and fake directories to collect frames
        for label, video_dir in enumerate(self.video_dirs):
            for video in os.listdir(video_dir):
                video_path = os.path.join(video_dir, video)
                for frame in os.listdir(video_path):
                    self.frames.append((os.path.join(video_path, frame), label))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path, label = self.frames[idx]
        image = Image.open(frame_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define your transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to ResNet's input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet mean/std normalization
])

# Load the dataset and split it into training and test sets
dataset = VideoFrameDataset('ProcessedFrames', transform=transform)
train_size = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])

# Create DataLoader for both training and testing sets
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# Load pre-trained ResNet model and modify the final layer for binary classification
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # You can use resnet34, resnet50, etc.
        
        # Modify the final fully connected layer to output 2 classes (real and fake)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the device (GPU or CPU)
model = ResNetModel().to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop with 15 epochs and progress bar
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for videos, labels in progress_bar:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    # Calculate accuracy for the epoch
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save the model after training
torch.save(model.state_dict(), "resnet_model.pth")

# Evaluate the model on the test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for videos, labels in test_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# Print the classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['real', 'fake']))

# Accuracy for the test set
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real', 'fake'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()