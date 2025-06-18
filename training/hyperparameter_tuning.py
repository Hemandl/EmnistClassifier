import optuna
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import EMNISTBalanced
from models.base_classifier import BaseClassifier
from models.category_classifier import CategoryClassifier
from models.modular_classifier import ModularClassifier
from training.trainer import Trainer

# Optimized values for efficient tuning
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 300
MAX_EPOCHS = 5


def objective(trial, model_type='modular', batch_size=256):
    """Optuna objective function with progress tracking and fixes"""
    # Common hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4)

    freeze_layers = None
    if model_type in ['base', 'modular']:
        freeze_layers = trial.suggest_int('freeze_layers', 3, 8)

    # Model-specific hyperparameters
    if model_type == 'modular':
        temp_class = trial.suggest_float('temp_class', 0.8, 1.2)
        temp_cat = trial.suggest_float('temp_cat', 0.8, 1.2)
        alpha = trial.suggest_float('alpha', 0.4, 0.6)

    # Create datasets
    train_set = EMNISTBalanced(split='train', samples_per_class=TRAIN_SAMPLES, augment=True)
    val_set = EMNISTBalanced(split='val', samples_per_class=VAL_SAMPLES, augment=False)

    # Calculate safe number of workers
    num_cpu = os.cpu_count()
    num_workers = min(8, num_cpu - 1) if num_cpu and num_cpu > 1 else 0

    # Create data loaders with async loading
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'modular':
        base_model = BaseClassifier(num_classes=36, freeze_layers=freeze_layers)
        category_model = CategoryClassifier()
        model = ModularClassifier(
            base_model,
            category_model,
            temp_class=temp_class,
            temp_cat=temp_cat,
            alpha=alpha
        )
    elif model_type == 'base':
        model = BaseClassifier(num_classes=36, freeze_layers=freeze_layers)
    else:
        model = CategoryClassifier()

    model.to(device)

    # Clear GPU cache if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=MAX_EPOCHS
    )

    # Train and validate with progress tracking
    best_acc = 0.0
    for epoch in range(MAX_EPOCHS):
        trainer.train_epoch(epoch)
        _, val_acc = trainer.validate()

        # Report intermediate accuracy
        trial.report(val_acc, epoch)

        # Prune unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


def optimize_hyperparameters(model_type='modular', n_trials=20, timeout=1200, batch_size=256):
    """
    Run hyperparameter optimization with progress tracking
    """
    # Create study with storage for resume capability
    storage_name = f"sqlite:///optuna_{model_type}.db"
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=storage_name,
        load_if_exists=True,
        study_name=f"emnist_{model_type}"
    )

    # Display study information
    print(f"Study name: emnist_{model_type}")
    print(f"Storage: {storage_name}")
    print(f"Number of existing trials: {len(study.trials)}")

    # Optimize with progress bar
    with tqdm(total=n_trials, desc="Hyperparameter Optimization") as pbar:
        def update_pbar(study, trial):
            if trial:
                pbar.update(1)
                current_value = trial.value or 0
                pbar.set_postfix({
                    "Current": f"{current_value:.4f}",
                    "Best": f"{study.best_value:.4f}"
                })

        study.optimize(
            lambda trial: objective(trial, model_type, batch_size),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[update_pbar],
            show_progress_bar=False  # We're using our own progress bar
        )

    # Save best parameters
    best_params = study.best_params
    with open(f'best_{model_type}_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest {model_type} accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Save study for future reference
    df = study.trials_dataframe()
    df.to_csv(f'{model_type}_optuna_trials.csv', index=False)

    return best_params