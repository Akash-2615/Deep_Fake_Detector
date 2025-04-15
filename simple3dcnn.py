import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch.nn as nn  # Import nn module here
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.video_paths = []
        self.labels = []

        for label_idx, label in enumerate(['real', 'fake']):
            label_dir = os.path.join(root_dir, label)
            for video_name in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_name)
                if os.path.isdir(video_path):
                    self.video_paths.append(video_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:self.num_frames]

        frames = []
        for f in frame_files:
            img_path = os.path.join(video_path, f)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (112, 112))
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
            frames.append(img)

        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))  # Padding

        video_tensor = torch.stack(frames, dim=1)  # Shape: [3, T, H, W]
        return video_tensor, label


import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.apool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Force output to fixed size
        self.fc1 = nn.Linear(64, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, D/2, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, D/4, H/4, W/4]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, D/8, H/8, W/8]
        x = self.apool(x)  # [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 64]
        x = self.fc1(x)  # [B, 256]
        return x


# Load dataset
dataset = VideoFrameDataset('ProcessedFrames')
train_size = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN().to(device)
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

# Final training accuracy after the last epoch
final_train_acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(device)
        outputs = model(videos)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['real', 'fake']))

# Accuracy for the test set
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Final test accuracy after evaluation
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real', 'fake'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save model
torch.save(model.state_dict(), "simple3dcnn_model.pth")