import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Dataset for Video Frames
class VideoFrameRNNWithCNN_Dataset(Dataset):
    def __init__(self, data_dir, frame_size=(224, 224), max_frames=16, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory containing 'real' and 'fake' folders.
            frame_size (tuple): Size to which each frame is resized (default: (224, 224)).
            max_frames (int): Maximum number of frames per video sequence (default: 16).
            transform (callable, optional): Optional transform to be applied to each frame.
        """
        self.data_dir = data_dir
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.transform = transform

        # Read the video folders (e.g., 'real', 'fake')
        self.video_folders = ['real', 'fake']
        self.video_paths = []
        for label in self.video_folders:
            label_path = os.path.join(data_dir, label)
            for video in os.listdir(label_path):
                video_folder_path = os.path.join(label_path, video)
                if os.path.isdir(video_folder_path):
                    self.video_paths.append((video_folder_path, label))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        
        frames = self._load_frames(video_path)

        # Check if frames are empty
        if not frames:
            frames = [np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)] * self.max_frames  # Pad with empty frames
        
        # Pad or truncate the frames to `max_frames`
        if len(frames) < self.max_frames:
            # Pad frames with the last frame
            padding = self.max_frames - len(frames)
            frames = frames + [frames[-1]] * padding  # Pad with last frame
        else:
            # Truncate frames
            frames = frames[:self.max_frames]
        
        frames = np.array(frames)

        # Apply transformations to each frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Convert the list of frames to a tensor
        frames = torch.stack(frames)  # Stack the frames into a single tensor

        # Return frames and label (encoded as 0 for real, 1 for fake)
        label = 0 if label == 'real' else 1
        return frames, label

    def _load_frames(self, video_folder_path):
        frames = []
        for frame_file in sorted(os.listdir(video_folder_path)):  # Ensure frames are loaded in order
            frame_path = os.path.join(video_folder_path, frame_file)
            if frame_path.endswith('.jpg') or frame_path.endswith('.png'):  # Assuming frames are stored as images
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        return frames

# Define the Model (CNN + RNN)
class VideoFrameRNNModel(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super(VideoFrameRNNModel, self).__init__()

        # CNN (ResNet)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        
        # RNN (LSTM)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Pass each frame through ResNet (CNN)
        resnet_out = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # Select frame i in the sequence
            resnet_out.append(self.resnet(frame))
        resnet_out = torch.stack(resnet_out, dim=1)  # Stack all the frame features
        
        # Pass the CNN features through LSTM
        lstm_out, _ = self.lstm(resnet_out)  # LSTM output (batch_size, seq_len, hidden_size)
        
        # Get the output from the last time step
        last_hidden_state = lstm_out[:, -1, :]  # Take the last time step output
        out = self.fc(last_hidden_state)
        
        return out

# Model initialization
model = VideoFrameRNNModel().to(device)

# Data transformation (resize and convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare dataset and dataloaders
data_dir = 'ProcessedFrames'  # Replace with your dataset directory
dataset = VideoFrameRNNWithCNN_Dataset(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# To store loss and accuracy for visualization
train_losses = []
train_accuracies = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for sequences, labels in progress_bar:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix(loss=running_loss / (total_preds / 4), accuracy=100 * correct_preds / total_preds)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_preds / total_preds
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), 'deepfake_rnn_model.pth')
print("Model saved to deepfake_rnn_model.pth")

# Evaluate the model on the test set
model.eval()
correct_preds = 0
total_preds = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_acc = 100 * correct_preds / total_preds
print(f"Test Accuracy: {test_acc:.2f}%")

# Generate Classification Report and Confusion Matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot the loss and accuracy curves
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()