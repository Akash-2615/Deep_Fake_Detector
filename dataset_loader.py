import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class RealDeepfakeDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self._prepare_dataset()

    def _prepare_dataset(self):
        for label_str in ['real', 'fake']:
            label_dir = os.path.join(self.root_dir, label_str)
            label = 0 if label_str == 'real' else 1
            for video_folder in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_folder)
                if os.path.isdir(video_path):
                    frame_files = sorted([
                        os.path.join(video_path, f)
                        for f in os.listdir(video_path)
                        if f.endswith(('.jpg', '.png'))
                    ])
                    if len(frame_files) >= self.sequence_length:
                        self.samples.append((frame_files, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        selected_frames = frame_paths[:self.sequence_length]

        # Single image for 2D CNN
        image = self.transform(Image.open(selected_frames[0]).convert('RGB'))

        # Stack for 3D CNN (C, D, H, W)
        video = [self.transform(Image.open(f).convert('RGB')) for f in selected_frames]
        video = torch.stack(video, dim=1)  # Shape: (C, D, H, W)

        # Flattened sequence for RNN (T, C*H*W)
        sequence = [self.transform(Image.open(f).convert('RGB')).flatten() for f in selected_frames]
        sequence = torch.stack(sequence)  # Shape: (T, C*H*W)

        return {
            'image': image,
            'video': video,
            'sequence': sequence,
            'label': torch.tensor(label, dtype=torch.long)
        }
