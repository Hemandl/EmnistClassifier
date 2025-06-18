import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class BaseClassifier(nn.Module):
    """Base classifier using ResNet18 with transfer learning"""

    def __init__(self, num_classes=36, freeze_layers=8):
        super().__init__()

        # Load pre-trained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.base = models.resnet18(weights=weights)

        # Adapt for grayscale input
        original_conv1 = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Copy grayscale weights
        if weights is not None:
            with torch.no_grad():
                # Use perceptual RGB to grayscale conversion
                rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=original_conv1.weight.device)
                new_weights = (original_conv1.weight.data * rgb_weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
                self.base.conv1.weight.data.copy_(new_weights)

        # Replace final layer
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, num_classes)

        # Freeze layers
        self.freeze_layers(freeze_layers)

    def freeze_layers(self, num_layers):
        """Freeze first num_layers of the model"""
        if num_layers <= 0:
            return

        layers = list(self.base.children())
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Input: (B, 1, H, W) Grayscale images
           Output: (B, C) Raw logits"""
        return self.base(x)