import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import EfficientNet_B4_Weights
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from pathlib import Path
from PIL import Image
from multiprocessing import freeze_support

# =========================
# 1. CONFIG
# =========================
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
VAL_SPLIT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

AI_DIR = "Ai_generated_dataset"
REAL_DIR = "real_dataset"

# =========================
# 2. TRANSFORMS (Improved)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.1),                 # NEW
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # NEW
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 3. DATASET
# =========================
class BinaryImageDataset(Dataset):
    def __init__(self, ai_root, real_root, transform=None):
        self.transform = transform
        self.samples = []

        self._collect_samples(ai_root, label=0)
        self._collect_samples(real_root, label=1)

    def _collect_samples(self, root_dir, label):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        root = Path(root_dir)

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_exts:
                self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        return image, label


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label

# =========================
# 4. MAIN
# =========================
def main():
    base_data = BinaryImageDataset(AI_DIR, REAL_DIR)

    if len(base_data) == 0:
        raise ValueError("No images found!")

    # Split dataset
    val_size = int(len(base_data) * VAL_SPLIT)
    val_size = max(1, min(val_size, len(base_data) - 1))
    train_size = len(base_data) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(base_data, [train_size, val_size], generator=generator)

    # Apply transforms
    train_data = TransformSubset(train_subset, train_transform)
    val_data = TransformSubset(val_subset, val_transform)

    # =========================
    # 🔥 CLASS IMBALANCE HANDLING
    # =========================
    train_labels = [base_data.samples[i][1] for i in train_subset.indices]

    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[label] for label in train_labels]

    sampler = WeightedRandomSampler(weights, len(weights))

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=sampler,   # instead of shuffle
        num_workers=4,
        pin_memory=USE_CUDA
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=USE_CUDA
    )

    print("Classes: ['ai_generated', 'real']")

    # =========================
    # 5. MODEL
    # =========================
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),   # NEW
        nn.Linear(in_features, 2)
    )

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    # =========================
    # 6. LOSS + OPTIMIZER
    # =========================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # NEW

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 🔥 Cosine Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

    # =========================
    # 7. TRAINING LOOP
    # =========================
    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # 🔥 Gradient Clipping (NEW)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / total

        # 🔥 Scheduler step (fixed)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
            print("✅ Best model saved!")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print(f"\n🔥 Training Complete! Best Accuracy: {best_acc:.2f}%")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    freeze_support()
    main()