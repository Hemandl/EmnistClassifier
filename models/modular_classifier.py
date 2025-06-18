import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularClassifier(nn.Module):
    """Modular classifier combining base and category classifiers"""

    def __init__(self, base_classifier, category_classifier, temp_class=1.0, temp_cat=1.0, alpha=0.5):
        super().__init__()
        self.base_classifier = base_classifier
        self.category_classifier = category_classifier

        # Fixed parameters (not learnable)
        self.temp_class = temp_class
        self.temp_cat = temp_cat
        self.alpha = alpha

    def forward(self, x):
        # Get outputs from both classifiers
        class_logits = self.base_classifier(x)
        cat_logits = self.category_classifier(x)

        # Apply temperature-scaled softmax
        class_probs = F.softmax(class_logits / self.temp_class, dim=1)
        cat_probs = F.softmax(cat_logits / self.temp_cat, dim=1)

        # Create category mapping
        weights = self._create_weight_matrix(cat_probs)

        # Fuse probabilities
        log_fused = self.alpha * torch.log(class_probs) + (1 - self.alpha) * torch.log(weights)
        fused_probs = torch.exp(log_fused - torch.logsumexp(log_fused, dim=1, keepdim=True))
        return torch.log(fused_probs + 1e-8)

    def _create_weight_matrix(self, cat_probs):
        """Map category probabilities to class weights"""
        # Define mapping: 0-9: digits (cat 2), 10-22: uppercase (cat 0), 23-35: lowercase (cat 1)
        cat_indices = torch.tensor(
            [2] * 10 + [0] * 13 + [1] * 13,
            device=cat_probs.device
        )

        return cat_probs[:, cat_indices]
