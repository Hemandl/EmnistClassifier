from torchvision.transforms import Compose, RandomApply, RandomAffine
from torchvision.transforms import RandomPerspective, ColorJitter, GaussianBlur


class EMNISTAugmenter:
    """EMNIST-specific augmentation pipeline with serializable transforms"""

    def __init__(self):
        self.transform = Compose([
            # Geometric transformations
            RandomApply([
                RandomAffine(
                    degrees=(-10, 10),  # Limited rotation
                    translate=(0.05, 0.05),  # Small translation
                    scale=(0.95, 1.05)  # Slight scaling
                ),
                RandomPerspective(distortion_scale=0.1, p=0.5)  # Mild perspective
            ], p=0.7),

            # Photometric transformations
            RandomApply([
                ColorJitter(brightness=0.1, contrast=0.1),  # Subtle variations
            ], p=0.4),

            # Noise and blur
            GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))  # Mild blur
        ])

    def __call__(self, img):
        return self.transform(img)