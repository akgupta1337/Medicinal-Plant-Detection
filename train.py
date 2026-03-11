from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# --- Config (change these values directly if you want) ---
DATA_DIR = "dataset"  # root folder (should contain train/ and optionally val/)
MODEL_PATH = "model/medicinal_plant_classifier.pth"
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2  # used only when there is no val/ directory


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Data preparation ---
    # We assume images are organized as: dataset/train/<class>/*.jpg
    train_dir = f"{DATA_DIR}/train"
    val_dir = f"{DATA_DIR}/val"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    if Path(val_dir).exists():
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    else:
        # If the user did not create a val/ folder, split the train set
        val_size = int(len(train_dataset) * VAL_SPLIT)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        # Ensure validation uses the correct transform
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine class names (works for ImageFolder and subsets of it)
    if hasattr(train_dataset, "classes"):
        class_names = train_dataset.classes
    else:
        class_names = train_dataset.dataset.classes

    # --- Model ---
    # Simple transfer learning: MobileNetV2 with a new final layer
    # Use the modern `weights=` argument (avoids deprecated warnings)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    num_classes = len(class_names)
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)

    print(f"Dataset has {num_classes} classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}; Val samples: {len(val_loader.dataset)}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, MODEL_PATH)
            print(f"Saved best model (val_acc={best_val_acc:.3f}) to {MODEL_PATH}")

    print("Training finished.")


if __name__ == "__main__":
    main()
