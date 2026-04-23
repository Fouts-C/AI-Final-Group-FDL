import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

class HurricaneDamageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Hurricane Image Data.
    """
    def __init__(self, csv_file: str, mask_dir: str, transform=None, label_encoder=None):
        """
        Args:
            csv_file (str): Path to the csv file with labels.
            mask_dir (str): Directory with all the mask images.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_encoder (LabelEncoder, optional): An existing sklearn LabelEncoder. 
                                                    If None, it fits a new one.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        self.data_frame = self.data_frame.dropna(subset=['Label_1'])
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.data_frame['encoded_label'] = self.label_encoder.fit_transform(self.data_frame['Label_1'].astype(str))
        else:
            self.label_encoder = label_encoder
            self.data_frame['encoded_label'] = self.label_encoder.transform(self.data_frame['Label_1'].astype(str))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.mask_dir / self.data_frame.iloc[idx]['Mask_Filename']
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label = self.data_frame.iloc[idx]['encoded_label']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(project_root: str, batch_size=32):
    """
    Returns train, validation, and test PyTorch dataloaders.
    """
    train_csv = Path(project_root) / 'data' / 'processed' / 'hurricane_train_labels.csv'
    val_csv = Path(project_root) / 'data' / 'processed' / 'hurricane_val_labels.csv'
    test_csv = Path(project_root) / 'data' / 'processed' / 'hurricane_test_labels.csv'
    
    mask_dir = Path(project_root) / 'data' / 'raw' / 'hurricane_damage' / 'MASK'
    
    # 1. Transforms for Training 
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # 2. Transforms for Validation/Testing
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Create Datasets
    train_dataset = HurricaneDamageDataset(train_csv, mask_dir, transform=train_transforms)
    val_dataset = HurricaneDamageDataset(val_csv, mask_dir, transform=eval_transforms, label_encoder=train_dataset.label_encoder)
    test_dataset = HurricaneDamageDataset(test_csv, mask_dir, transform=eval_transforms, label_encoder=train_dataset.label_encoder)

    # 4. Create Dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Classes mapped by LabelEncoder: {list(train_dataset.label_encoder.classes_)}")
    
    return train_loader, val_loader, test_loader, len(train_dataset.label_encoder.classes_)

# Quick Test Block
if __name__ == '__main__':
    root = Path(__file__).parent.parent
    train_dl, val_dl, test_dl, num_classes = get_dataloaders(root, batch_size=4)
    images, labels = next(iter(train_dl))
    print(f"Loaded a batch of {len(images)} images.")
    print(f"Image tensor shape: {images.shape}")
    print(f"Labels: {labels}")
