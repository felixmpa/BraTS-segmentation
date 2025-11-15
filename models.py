import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import numpy as np

class BasicCNN(nn.Module):
    """Basic CNN for brain tumor segmentation"""
    def __init__(self, in_channels=4, num_classes=5):
        super(BasicCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    """U-Net implementation for medical image segmentation"""
    def __init__(self, in_channels=4, num_classes=5):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        
        # Encoder
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        
        # Decoder
        self.up6 = up_conv(1024, 512)
        self.conv6 = conv_block(1024, 512)
        self.up7 = up_conv(512, 256)
        self.conv7 = conv_block(512, 256)
        self.up8 = up_conv(256, 128)
        self.conv8 = conv_block(256, 128)
        self.up9 = up_conv(128, 64)
        self.conv9 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        x1 = self.maxpool(conv1)
        
        conv2 = self.conv2(x1)
        x2 = self.maxpool(conv2)
        
        conv3 = self.conv3(x2)
        x3 = self.maxpool(conv3)
        
        conv4 = self.conv4(x3)
        x4 = self.maxpool(conv4)
        
        conv5 = self.conv5(x4)
        
        # Decoder
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        
        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        
        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        
        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        
        output = self.final(conv9)
        return output

class TransferLearningUNet(nn.Module):
    """U-Net with pre-trained encoder for transfer learning"""
    def __init__(self, encoder_name='resnet50', in_channels=4, num_classes=5, encoder_weights=None):
        super(TransferLearningUNet, self).__init__()
        
        # Use segmentation_models_pytorch for transfer learning
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)

class ResNetUNet(nn.Module):
    """ResNet-based U-Net with transfer learning"""
    def __init__(self, in_channels=4, num_classes=5):
        super(ResNetUNet, self).__init__()
        
        # Pre-trained ResNet50 encoder
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify first conv layer for 4-channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize new conv layer
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Encoder layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Decoder
        self.up_conv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv_up4 = self._make_conv_block(2048, 1024)
        
        self.up_conv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up3 = self._make_conv_block(1024, 512)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = self._make_conv_block(512, 256)
        
        self.up_conv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.conv_up1 = self._make_conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1_pool = self.maxpool(x1)
        
        x2 = self.layer1(x1_pool)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        # Decoder with skip connections
        up4 = self.up_conv4(x5)
        merge4 = torch.cat([up4, x4], dim=1)
        up4 = self.conv_up4(merge4)
        
        up3 = self.up_conv3(up4)
        merge3 = torch.cat([up3, x3], dim=1)
        up3 = self.conv_up3(merge3)
        
        up2 = self.up_conv2(up3)
        merge2 = torch.cat([up2, x2], dim=1)
        up2 = self.conv_up2(merge2)
        
        up1 = self.up_conv1(up2)
        # Ensure spatial dimensions match for concatenation
        if up1.shape[2:] != x1.shape[2:]:
            up1 = F.interpolate(up1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        merge1 = torch.cat([up1, x1], dim=1)
        up1 = self.conv_up1(merge1)
        
        output = self.final_conv(up1)
        # Ensure output matches input dimensions
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

class AttentionUNet(nn.Module):
    """U-Net with attention mechanisms"""
    def __init__(self, in_channels=4, num_classes=5):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.conv1 = self._conv_block(in_channels, 64)
        self.conv2 = self._conv_block(64, 128)
        self.conv3 = self._conv_block(128, 256)
        self.conv4 = self._conv_block(256, 512)
        self.conv5 = self._conv_block(512, 1024)
        
        # Decoder with attention
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = self._attention_block(512, 512, 256)
        self.conv6 = self._conv_block(1024, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = self._attention_block(256, 256, 128)
        self.conv7 = self._conv_block(512, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = self._attention_block(128, 128, 64)
        self.conv8 = self._conv_block(256, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = self._attention_block(64, 64, 32)
        self.conv9 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, num_classes, 1)
        self.maxpool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _attention_block(self, F_g, F_l, F_int):
        return nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.Conv2d(F_l, F_int, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        x1 = self.maxpool(conv1)
        
        conv2 = self.conv2(x1)
        x2 = self.maxpool(conv2)
        
        conv3 = self.conv3(x2)
        x3 = self.maxpool(conv3)
        
        conv4 = self.conv4(x3)
        x4 = self.maxpool(conv4)
        
        conv5 = self.conv5(x4)
        
        # Decoder with attention
        up6 = self.up6(conv5)
        att6 = self.att6(up6)
        conv4_att = conv4 * att6
        merge6 = torch.cat([up6, conv4_att], dim=1)
        conv6 = self.conv6(merge6)
        
        up7 = self.up7(conv6)
        att7 = self.att7(up7)
        conv3_att = conv3 * att7
        merge7 = torch.cat([up7, conv3_att], dim=1)
        conv7 = self.conv7(merge7)
        
        up8 = self.up8(conv7)
        att8 = self.att8(up8)
        conv2_att = conv2 * att8
        merge8 = torch.cat([up8, conv2_att], dim=1)
        conv8 = self.conv8(merge8)
        
        up9 = self.up9(conv8)
        att9 = self.att9(up9)
        conv1_att = conv1 * att9
        merge9 = torch.cat([up9, conv1_att], dim=1)
        conv9 = self.conv9(merge9)
        
        output = self.final(conv9)
        return output

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ for semantic segmentation"""
    def __init__(self, in_channels=4, num_classes=5):
        super(DeepLabV3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)

class VariationalAutoencoder(nn.Module):
    """VAE for brain tumor segmentation with latent space learning"""
    def __init__(self, in_channels=4, num_classes=5, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 15 * 15, latent_dim)
        self.fc_logvar = nn.Linear(512 * 15 * 15, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 15 * 15)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 3, stride=2, padding=1, output_padding=1),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 15, 15)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def get_model(model_name: str, in_channels: int = 4, num_classes: int = 5, **kwargs):
    """Factory function to create models"""
    models_dict = {
        'basic_cnn': BasicCNN,
        'unet': UNet,
        'resnet_unet': ResNetUNet,
        'attention_unet': AttentionUNet,
        'deeplabv3plus': DeepLabV3Plus,
        'vae': VariationalAutoencoder,
        'transfer_unet': TransferLearningUNet,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name](in_channels=in_channels, num_classes=num_classes, **kwargs)

# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * ce + (1 - self.alpha) * dice