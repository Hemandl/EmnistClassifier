import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from data.augmentation import EMNISTAugmenter


# Named functions replace lambdas for better serialization
def rotate_90(img):
    return img.rotate(-90)


def flip_horizontal(img):
    return img.transpose(0)


class EMNISTBalanced(Dataset):
    """Balanced EMNIST dataset for 36 classes with serializable transforms"""

    CLASS_OFFSETS = {
        'digits': 0,
        'uppercase': 10,
        'lowercase': 23
    }

    CLASS_COUNTS = {
        'digits': 10,
        'uppercase': 13,
        'lowercase': 13
    }

    def __init__(self, split='train', samples_per_class=5000, augment=False):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val' or 'test'")

        self.split = split
        self.samples_per_class = samples_per_class
        self.augment = augment
        self.rng = np.random.RandomState(42)

        # Download and load EMNIST dataset
        self.dataset = EMNIST(
            root='./data',
            split='byclass',
            train=(split != 'test'),
            download=True
        )

        # Filter for our 36 classes
        self.indices, self.targets = self._filter_classes()

        # Balance dataset
        self.balanced_indices = self._balance_dataset()

        # Setup transformations
        self.transform = self._get_transforms()

    def _filter_classes(self):
        """Filter dataset for 36 classes"""
        indices = []
        targets = []

        for idx, (_, label) in enumerate(self.dataset):
            if label < 36:  # Keep only our 36 classes
                indices.append(idx)
                targets.append(label)

        return indices, torch.tensor(targets)

    def _balance_dataset(self):
        """Balance classes with oversampling"""
        class_indices = {i: [] for i in range(36)}

        # Group indices by class
        for idx, label in zip(self.indices, self.targets):
            class_indices[label.item()].append(idx)

        balanced_indices = []

        # Sample balanced dataset
        for cls in range(36):
            indices = class_indices[cls]
            n_samples = len(indices)

            if n_samples >= self.samples_per_class:
                # Select subset
                selected = self.rng.choice(indices, self.samples_per_class, replace=False)
            else:
                # Oversample with replacement
                selected = list(indices)
                n_needed = self.samples_per_class - n_samples
                extra = self.rng.choice(indices, n_needed, replace=True)
                selected.extend(extra)

            balanced_indices.extend(selected)

        return balanced_indices

    def _get_transforms(self):
        """Get serializable transforms"""
        base_transforms = [
            # EMNIST specific corrections using named functions
            rotate_90,  # Correct orientation
            flip_horizontal,  # Flip horizontally
            ToTensor(),
            Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ]

        if self.split == 'train' and self.augment:
            # Insert augmentation at the beginning
            return Compose([EMNISTAugmenter()] + base_transforms)
        else:
            return Compose(base_transforms)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        """Get image and label with applied transforms"""
        dataset_idx = self.balanced_indices[idx]
        image, label = self.dataset[dataset_idx]

        if self.transform:
            image = self.transform(image)

        return image, label