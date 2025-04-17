import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as F
import re
import time
from torchvision.models.video import r3d_18
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# -------------------- Dataset --------------------


class FinchFrameDataset(Dataset):
    def __init__(self, frames_dir, clip_length=16, insc_fraction=0.2):
        self.frames_dir = frames_dir
        self.clip_length = clip_length
        self.insc_fraction = insc_fraction

        self.class_folders = [d for d in os.listdir(
            frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
        self.label2idx = {label: idx for idx,
                          label in enumerate(self.class_folders)}

        self.samples = []

        for label in self.class_folders:
            label_dir = os.path.join(frames_dir, label)
            video_groups = {}
            for fname in os.listdir(label_dir):
                match = re.match(r'(\d+)_frame_\d+\.jpg', fname)
                if match:
                    video_num = match.group(1)
                    if video_num not in video_groups:
                        video_groups[video_num] = []
                    full_path = os.path.join(label_dir, fname)
                    video_groups[video_num].append(full_path)

            clip_items = []
            for video_num, frame_paths in video_groups.items():
                frame_paths = sorted(frame_paths, key=lambda x: int(
                    re.search(r'_frame_(\d+)', x).group(1)))
                clip_items.append((frame_paths, self.label2idx[label]))

            if label == "INSC":
                clip_items = clip_items[:int(
                    len(clip_items) * self.insc_fraction)]

            self.samples.extend(clip_items)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []

        for i in range(self.clip_length):
            if i < len(frame_paths):
                frame = Image.open(frame_paths[i]).convert('RGB')
                frame = F.resize(frame, 112)
                frame = F.to_tensor(frame)
                frame = F.normalize(frame, mean=[0.45, 0.45, 0.45], std=[
                                    0.225, 0.225, 0.225])
                frames.append(frame)
            else:
                frames.append(frames[-1])

        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        return frames, torch.tensor(label).long()


# -------------------- Model --------------------
class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DClassifier, self).__init__()
        self.backbone = r3d_18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# -------------------- Training --------------------
if __name__ == "__main__":
    frames_dir = r"E:\Multicategory_frames"
    dataset = FinchFrameDataset(frames_dir, clip_length=16)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label2idx)
    model = ResNet3DClassifier(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 🟢 Preload 1 batch BEFORE tqdm loop to avoid ETA freeze
    next(iter(dataloader))

    for epoch in range(1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} finished in {elapsed:.2f} seconds | Total Loss: {running_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

# Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=list(dataset.label2idx.keys())))

# Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.label2idx.keys(), yticklabels=dataset.label2idx.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
