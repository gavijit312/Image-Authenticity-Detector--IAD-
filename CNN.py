import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import EfficientNet_B4_Weights
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from pathlib import Path
from PIL import Image
from multiprocessing import freeze_support


BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
VAL_SPLIT = 0.2
IMG_SIZE = 224
CLASSIFIER_DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

AI_DIR = "Ai_generated_dataset"
REAL_DIR = "real_dataset"


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class BinaryImageDataset(Dataset):
    def __init__(self, ai_root, real_root):
        self.samples = []
        self._collect(ai_root, 0)
        self._collect(real_root, 1)

    def _collect(self, root_dir, label):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in Path(root_dir).rglob("*"):
            if path.suffix.lower() in valid_exts:
                self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # skip bad image
            return self.__getitem__((idx + 1) % len(self))
        return image, label


class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label



def main():

    base_data = BinaryImageDataset(AI_DIR, REAL_DIR)

    if len(base_data) == 0:
        raise ValueError("No images found!")

    print(f"Total images: {len(base_data)}")

   
    val_size = int(len(base_data) * VAL_SPLIT)
    val_size = max(1, min(val_size, len(base_data) - 1))
    train_size = len(base_data) - val_size

    train_subset, val_subset = random_split(
        base_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_data = TransformDataset(train_subset, train_transform)
    val_data = TransformDataset(val_subset, val_transform)

    
    train_labels = [base_data.samples[i][1] for i in train_subset.indices]

    class_counts = [train_labels.count(0), train_labels.count(1)]
    print("Class distribution:", class_counts)

    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in class_counts],
        dtype=torch.float32,
        device=DEVICE,
    )

    
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  
        pin_memory=USE_CUDA
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, 
        pin_memory=USE_CUDA
    )

    print("Classes: ['ai_generated', 'real']")
    print("Training started...\n")

    
  
    
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(CLASSIFIER_DROPOUT),
        nn.Linear(in_features, 2)
    )

    model = model.to(DEVICE)

    
    

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler(enabled=USE_CUDA)

   
    
    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):

        
        model.train()
        running_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            if i % 50 == 0:
                print(f"Epoch {epoch+1} - Batch {i}")

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

       
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / total

        scheduler.step()

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

     
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
            print("✅ Best model saved!\n")
        else:
            counter += 1

        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print(f"\n Training Complete! Best Accuracy: {best_acc:.2f}%")




if __name__ == "__main__":
    freeze_support()
    main()