import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from HurricaneDamageDataset import get_dataloaders


# YOLO Nano CNN:
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + SiLU"""
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """3x3 3x3 residual unit used inside C2f."""
    def __init__(self, ch, shortcut=True):
        super().__init__()
        self.cv1 = ConvBlock(ch, ch, 3)
        self.cv2 = ConvBlock(ch, ch, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        mid = out_ch // 2
        self.cv1         = ConvBlock(in_ch, 2 * mid, 1)
        self.cv2         = ConvBlock((2 + n) * mid, out_ch, 1)
        self.bottlenecks = nn.ModuleList(Bottleneck(mid, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        mid       = in_ch // 2
        self.cv1  = ConvBlock(in_ch, mid, 1)
        self.cv2  = ConvBlock(mid * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# YOLO Nano classifier
class YOLONanoClassifier(nn.Module):
    """ YOLOv8-nano for image classification."""
    def __init__(self, num_classes: int):
        super().__init__()

        self.stem   = ConvBlock(3,   16,  3, 2)
        self.stage1 = nn.Sequential(ConvBlock(16,  32,  3, 2), C2f(32,  32,  1))
        self.stage2 = nn.Sequential(ConvBlock(32,  64,  3, 2), C2f(64,  64,  2))
        self.stage3 = nn.Sequential(ConvBlock(64,  128, 3, 2), C2f(128, 128, 2))
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, 3, 2),
            C2f(256, 256, 1),
            SPPF(256, 256),
        )
        self.head = nn.Sequential(
            ConvBlock(256, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


# Training loop
def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=100, save_path=None):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss     = 0.0
            running_corrects = 0

            loader = dataloaders[phase]
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc  = running_corrects.float() / len(loader.dataset)
            print(f"  {phase.capitalize():5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"  ** New best model saved (val acc: {best_val_acc:.4f})")

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return model, history


def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'],   label='Val Accuracy')
    axes[1].axhline(y=1/6, color='gray', linestyle='--', label='Random Chance (16.7%)')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('YOLONanoClassifier — Hurricane Damage Training History', fontsize=13)
    plt.tight_layout()
    out_path = save_dir / 'training_history.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Training history plot saved to {out_path}")


def plot_confusion_matrix(model, loader, device, class_names, save_dir):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set\n(Hurricane Damage Level 0–5)', fontsize=13)
    plt.tight_layout()
    out_path = save_dir / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {out_path}")

    correct = np.trace(cm)
    total   = np.sum(cm)
    print(f"Test Accuracy: {correct}/{total} = {correct/total:.4f}")


# Entry point
if __name__ == "__main__":
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data
    project_root = Path(__file__).parent.parent
    train_dl, val_dl, test_dl, num_classes = get_dataloaders(project_root, batch_size=32)
    dataloaders = {'train': train_dl, 'val': val_dl}

    print(f"\nBuilding YOLONanoClassifier with {num_classes} output classes...")
    model = YOLONanoClassifier(num_classes=num_classes).to(device)

    train_csv = project_root / 'data' / 'processed' / 'hurricane_train_labels.csv'
    train_counts = pd.read_csv(train_csv)['Label_1'].value_counts().sort_index()
    total = train_counts.sum()
    weights = torch.tensor(
        [total / (num_classes * train_counts[c]) for c in range(num_classes)],
        dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    save_dir       = project_root / 'hurricane_damage_code'
    best_model_path = save_dir / 'yolo_nano_hurricane_best.pth'

    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=100, save_path=best_model_path
    )

    # Load best checkpoint for evaluation
    trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from {best_model_path}")

    # Visualizations
    plot_training_history(history, save_dir)

    class_names = [f'Level {i}' for i in range(num_classes)]
    plot_confusion_matrix(trained_model, test_dl, device, class_names, save_dir)
