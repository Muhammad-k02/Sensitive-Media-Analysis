import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class ViolenceEndToEndDataset(Dataset):
    def __init__(self, frame_paths, labels, transform=None):
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        label = self.labels[idx]

        image = Image.open(frame_path)
        if self.transform:
            image = self.transform(image)

        return image, label
