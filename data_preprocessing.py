import pandas as pd
import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Dict

class BraTSDataPreprocessor:
    def __init__(self, csv_path: str, target_size: Tuple[int, int] = (240, 240)):
        self.csv_path = csv_path
        self.target_size = target_size
        self.df = pd.read_csv(csv_path)
        
    def analyze_dataset(self) -> Dict:
        """Analyze dataset distribution and statistics"""
        analysis = {
            'total_slices': len(self.df),
            'unique_volumes': self.df['volume'].nunique(),
            'label_distribution': {
                'background': (self.df['background_ratio'] == 1.0).sum(),
                'with_tumor': (self.df['background_ratio'] < 1.0).sum()
            },
            'mean_background_ratio': self.df['background_ratio'].mean(),
            'tumor_presence_ratio': (self.df['background_ratio'] < 1.0).mean()
        }
        return analysis
    
    def create_balanced_dataset(self, tumor_ratio: float = 0.3) -> pd.DataFrame:
        """Create a balanced dataset with specified tumor ratio"""
        tumor_slices = self.df[self.df['background_ratio'] < 1.0]
        background_slices = self.df[self.df['background_ratio'] == 1.0]
        
        n_tumor = min(len(tumor_slices), int(len(tumor_slices) / tumor_ratio))
        n_background = n_tumor * int((1 - tumor_ratio) / tumor_ratio)
        
        balanced_tumor = tumor_slices.sample(n=min(n_tumor, len(tumor_slices)), random_state=42)
        balanced_background = background_slices.sample(n=min(n_background, len(background_slices)), random_state=42)
        
        balanced_df = pd.concat([balanced_tumor, balanced_background]).reset_index(drop=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

class BraTSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # For this implementation, we'll create synthetic data since actual H5 files aren't available
        # In real implementation, you would load from H5 files:
        # with h5py.File(row['slice_path'], 'r') as f:
        #     image = f['image'][:]  # Multi-modal MRI data
        #     mask = f['mask'][:]    # Segmentation mask
        
        # Synthetic data for demonstration (4-channel MRI: T1, T2, T1c, FLAIR)
        image = np.random.rand(4, 240, 240).astype(np.float32)
        
        # Create synthetic mask based on background_ratio
        mask = np.zeros((240, 240), dtype=np.float32)
        if row['background_ratio'] < 1.0:
            # Add synthetic tumor regions
            center_x, center_y = np.random.randint(60, 180, 2)
            radius = np.random.randint(10, 40)
            y, x = np.ogrid[:240, :240]
            mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[mask_circle] = np.random.choice([1, 2, 4], p=[0.5, 0.3, 0.2])  # Different tumor classes
        
        if self.transform:
            # Apply augmentations to each channel separately
            augmented_channels = []
            for i in range(4):
                augmented = self.transform(image=image[i], mask=mask)
                augmented_channels.append(augmented['image'])
            image = np.stack(augmented_channels, axis=0)
            mask = augmented['mask']
        
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class DataAugmentationPipeline:
    @staticmethod
    def get_training_transforms():
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05, shift_limit=0.05),
            A.GaussNoise(p=0.3),
            A.Resize(240, 240),
        ])
    
    @staticmethod
    def get_validation_transforms():
        return A.Compose([
            A.Resize(240, 240),
        ])

def create_data_loaders(csv_path: str, batch_size: int = 8, val_split: float = 0.2):
    """Create training and validation data loaders"""
    preprocessor = BraTSDataPreprocessor(csv_path)
    
    # Analyze dataset
    analysis = preprocessor.analyze_dataset()
    print("Análisis del Dataset:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Create balanced dataset
    balanced_df = preprocessor.create_balanced_dataset()
    print(f"\nTamaño del dataset balanceado: {len(balanced_df)}")
    
    # Split data
    train_df, val_df = train_test_split(balanced_df, test_size=val_split, 
                                       stratify=balanced_df['target'], random_state=42)
    
    # Create datasets
    train_dataset = BraTSDataset(train_df, DataAugmentationPipeline.get_training_transforms(), mode='train')
    val_dataset = BraTSDataset(val_df, DataAugmentationPipeline.get_validation_transforms(), mode='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, analysis

def visualize_sample_data(data_loader):
    """Visualize sample data with augmentations"""
    data_iter = iter(data_loader)
    images, masks = next(data_iter)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Show different MRI modalities
        axes[0, i].imshow(images[0, i].numpy(), cmap='gray')
        axes[0, i].set_title(f'Modality {["T1", "T2", "T1c", "FLAIR"][i]}')
        axes[0, i].axis('off')
    
    # Show mask overlays
    for i in range(4):
        axes[1, i].imshow(images[0, i].numpy(), cmap='gray', alpha=0.7)
        mask_overlay = masks[0].numpy()
        axes[1, i].imshow(mask_overlay, cmap='jet', alpha=0.3)
        axes[1, i].set_title(f'Mask Overlay - {["T1", "T2", "T1c", "FLAIR"][i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizacion_datos_muestra.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    csv_path = "BraTS20 Training Metadata.csv"
    train_loader, val_loader, analysis = create_data_loaders(csv_path, batch_size=4)
    
    print(f"\nLotes de entrenamiento: {len(train_loader)}")
    print(f"Lotes de validación: {len(val_loader)}")
    
    # Visualize sample data
    visualize_sample_data(train_loader)