import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast


class Trainer:
    """Enhanced trainer with progress tracking and checkpointing"""

    def __init__(self, model, device, train_loader, val_loader,
                 lr=0.001, weight_decay=1e-4, max_epochs=30):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs

        # Korrekte Loss-Funktion für Raw-Logits
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Mixed precision training
        self.scaler = GradScaler(enabled=device.type == 'cuda')

        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def train_epoch(self, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Create progress bar
        batch_bar = tqdm(total=len(self.train_loader),
                         desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                         position=0,
                         leave=True)

        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            # Move data to device asynchronously
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision training
            with autocast(enabled=self.device.type == 'cuda', device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate metrics
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            batch_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{correct / total:.4f}"
            })
            batch_bar.update(1)

        # Calculate epoch metrics
        avg_loss = epoch_loss / total
        accuracy = correct / total
        epoch_time = time.time() - start_time

        batch_bar.close()
        return avg_loss, accuracy, epoch_time

    def validate(self):
        """Validate model with progress tracking"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validation", position=0, leave=False)
            for inputs, labels in val_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)  # Korrekte Vorhersage aus Logits
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix({
                    "Acc": f"{correct / total:.4f}"
                })

        avg_loss = val_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, save_path='best_model.pth'):
        """Train model with checkpointing"""
        print(f"Training for {self.max_epochs} epochs...")
        print(
            f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Time (s)':<8} | {'LR':<10}")
        print("-" * 80)

        for epoch in range(self.max_epochs):
            # Train and validate
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_acc)

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)

            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved at epoch {epoch + 1} with accuracy {val_acc:.4f}")

            # Print progress
            print(f"{epoch + 1:<6} | {train_loss:<10.4f} | {train_acc:<10.4f} | "
                  f"{val_loss:<10.4f} | {val_acc:<10.4f} | {epoch_time:<8.1f} | "
                  f"{self.optimizer.param_groups[0]['lr']:<10.2e}")

        print(f"\nTraining complete. Best validation accuracy: {self.best_acc:.4f} at epoch {self.best_epoch + 1}")
        return self.history