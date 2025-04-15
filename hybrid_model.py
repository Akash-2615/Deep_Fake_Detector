import os
import cv2
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
from torchvision.transforms import ToPILImage

class DeepFakeFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, img_size=112):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.data = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self._prepare_dataset()

    def _prepare_dataset(self):
        for label, category in enumerate(['real', 'fake']):
            class_dir = os.path.join(self.root_dir, category)
            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                if os.path.isdir(video_path):
                    frames = sorted(os.listdir(video_path))
                    if len(frames) >= 2:
                        self.data.append((video_path, frames))
                        self.labels.append(label)

    def _load_frames(self, video_path, frame_list):
        total = len(frame_list)
        if total >= self.num_frames:
            start = random.randint(0, total - self.num_frames)
            selected = frame_list[start:start + self.num_frames]
        else:
            selected = frame_list + [frame_list[-1]] * (self.num_frames - total)
        
        frames = []
        for fname in selected:
            img_path = os.path.join(video_path, fname)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            frames.append(img)
        return torch.stack(frames)  # Shape: [N, 3, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, frame_list = self.data[idx]
        frames_tensor = self._load_frames(video_path, frame_list)
        label = self.labels[idx]
        return frames_tensor, torch.tensor(label)
    

dataset = DeepFakeFrameDataset('ProcessedFrames', num_frames=16)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    frames, labels = batch
    print(frames.shape)  # (B, N, 3, 112, 112)
    print(labels)
    break

class ResNetCustom(nn.Module):
    def __init__(self, output_dim=256, dropout_rate=0.3, freeze_base=False):
        super(ResNetCustom, self).__init__()

        # Load deeper backbone
        resnet = models.resnet50(pretrained=True)
        
        # Optionally freeze layers
        if freeze_base:
            for param in resnet.parameters():
                param.requires_grad = False

        self.base = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.feature_dim = resnet.fc.in_features  # 2048 for ResNet50

        # Enhanced FC layer
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)                 # Flatten time
        features = self.base(x).squeeze()        # (B*T, 2048)
        features = self.head(features)           # (B*T, output_dim)
        features = features.view(B, T, -1)       # Reshape back: (B, T, output_dim)
        return features
    
class DeepFake3DCNN(nn.Module):
    def __init__(self, output_dim=256):
        super(DeepFake3DCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),  # (B, 3, T, H, W) -> (B, 32, T, H, W)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):  # x: (B, 3, T, H, W)
        x = self.conv_block(x)
        x = self.fc(x)
        return x
    
class DeepFakeBiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2, output_dim=128):
        super(DeepFakeBiLSTMWithAttention, self).__init__()

        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )

        # Attention weights
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):  # x: (B, T, input_dim)
        lstm_out, _ = self.bilstm(x)  # lstm_out: (B, T, 2*hidden_dim)

        # Attention weights for each timestep
        attn_weights = self.attention_fc(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T, 1)

        # Weighted sum of hidden states
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2*hidden_dim)

        output = self.output_fc(context)  # (B, output_dim)
        return output
    
class HybridDeepFakeDetector(nn.Module):
    def __init__(self):
        super(HybridDeepFakeDetector, self).__init__()
        self.resnet = ResNetCustom(output_dim=128)
        self.c3d = DeepFake3DCNN(output_dim=128)
        self.rnn = DeepFakeBiLSTMWithAttention(input_dim=128, hidden_dim=128, output_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # Binary classification (real/fake)
        )
    def forward(self, x):
    # x: (B, T, C, H, W)

    # ResNet path
     resnet_feat = self.resnet(x)  # (B, T, 128)

    # 3D CNN path - convert to (B, C, T, H, W)
     x_3d = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
     c3d_feat = self.c3d(x_3d)        # (B, 128)

    # BiLSTM with Attention path
     lstm_feat = self.rnn(resnet_feat)  # (B, 128)

    # Combine all features
     combined = torch.cat([c3d_feat, lstm_feat, resnet_feat.mean(dim=1)], dim=1)  # (B, 384)
     out = self.classifier(combined)
     return out
 
 
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, save_path="hybrid_model.pth", device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"\nTrain Loss: {train_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        print(f"Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")

        # Save best modela
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model to {save_path}")

    print("\n🎉 Training Completed.")

    
dataset = DeepFakeFrameDataset('ProcessedFrames', num_frames=16)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = HybridDeepFakeDetector()

train_model(model, train_loader, val_loader, num_epochs=10, save_path="hybrid_deepfake.pth")
    
model = HybridDeepFakeDetector()
model.load_state_dict(torch.load("hybrid_deepfake.pth"))
model.eval()

def evaluate_model(model, data_loader, class_names=["Real", "Fake"], device='cuda'):
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return all_labels, all_preds, all_probs


def plot_conf_matrix(y_true, y_pred, class_names=["Real", "Fake"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(y_true, y_probs):
    fake_probs = [p[1] for p in y_probs]  # prob of class "Fake"
    plt.figure(figsize=(8, 4))
    sns.histplot(fake_probs, bins=20, kde=True, color='purple')
    plt.title("Prediction Confidence for 'Fake' Class")
    plt.xlabel("Confidence (Probability)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs, pos_label=1):
    y_scores = [p[pos_label] for p in y_probs]
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("🔍 ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_bar(y_true, y_pred, class_names=["Real", "Fake"]):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, precision, width, label='Precision', color='skyblue')
    plt.bar(x + width/2, recall, width, label='Recall', color='salmon')
    plt.xticks(x, class_names)
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.title('📌 Per-Class Precision and Recall')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def preview_misclassified_frames(model, data_loader, device='cuda', max_preview=5):
    from torchvision.transforms import ToPILImage

    model.eval()
    model.to(device)
    shown = 0
    to_pil = ToPILImage()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(y)):
                if preds[i] != y[i]:
                    clip = x[i]  # (T, C, H, W) or (C, H, W) if already 4D
                    if clip.ndim == 4:
                        # Choose the middle frame
                        middle_frame = clip[len(clip) // 2]  # shape (C, H, W)
                    elif clip.ndim == 3:
                        middle_frame = clip  # already (C, H, W)
                    else:
                        continue  # skip if unexpected shape

                    img = to_pil(middle_frame.cpu())
                    plt.imshow(img)
                    plt.title(f"❌ Pred: {preds[i].item()} | ✅ Actual: {y[i].item()}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()

                    shown += 1
                    if shown >= max_preview:
                        return


# 🔍 Run Full Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_true, y_pred, y_prob = evaluate_model(model, val_loader, device=device)

# 🧮 Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")

# 📉 Plots
plot_conf_matrix(y_true, y_pred)
plot_confidence_distribution(y_true, y_prob)
plot_roc_curve(y_true, y_prob)
plot_precision_recall_bar(y_true, y_pred)

# 🖼️ Show Misclassified Frames
preview_misclassified_frames(model, val_loader, device=device)

    