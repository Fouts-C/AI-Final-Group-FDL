import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

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
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10):
    best_val_acc = 0.0

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
            epoch_acc  = running_corrects.double() / len(loader.dataset)
            print(f"  {phase.capitalize():5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return model


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

    trained_model = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10
    )

    model_save_path = project_root / 'hurricane_damage_code' / 'yolo_nano_hurricane.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
