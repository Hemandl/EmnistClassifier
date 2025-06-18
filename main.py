import argparse
import json
import os
import time

import numpy as np
import torch

from data.dataset import EMNISTBalanced
from evaluation.evaluate import evaluate_model
from models.base_classifier import BaseClassifier
from models.category_classifier import CategoryClassifier
from models.modular_classifier import ModularClassifier
from training.hyperparameter_tuning import optimize_hyperparameters
from training.trainer import Trainer


def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    # Setzen des Generators für reproduzierbare DataLoader-Ergebnisse
    return torch.Generator().manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description='EMNIST Classification')
    parser.add_argument('--model', type=str, choices=['base', 'modular', 'category'], default='modular',
                        help='Model type: base, modular or category')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed for reproducibility
    generator = set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")

    # Hyperparameter tuning
    if args.tune:
        print(f"\n{'=' * 50}")
        print(f"Starting hyperparameter tuning for {args.model} model")
        print(f"{'=' * 50}")
        start_time = time.time()
        best_params = optimize_hyperparameters(
            model_type=args.model,
            batch_size=args.batch_size
        )
        tuning_time = time.time() - start_time
        print(f"\nTuning completed in {tuning_time // 60:.0f}m {tuning_time % 60:.0f}s")
        print(f"Best parameters: {best_params}")

    # Model training
    if args.train:
        print(f"\n{'=' * 50}")
        print(f"Training {args.model} model")
        print(f"{'=' * 50}")

        # Load best parameters or use defaults
        param_file = f'best_{args.model}_params.json'
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                best_params = json.load(f)
            print(f"Loaded best parameters: {best_params}")
        else:
            print("Using default parameters")
            best_params = {
                'lr': 0.001,
                'weight_decay': 1e-4,
            }
            # Modellspezifische Standardparameter
            if args.model in ['base', 'modular']:
                best_params['freeze_layers'] = 8
            if args.model == 'modular':
                best_params.update({
                    'temp_class': 1.0,
                    'temp_cat': 1.0,
                    'alpha': 0.5
                })

        # Create datasets
        train_set = EMNISTBalanced(split='train', samples_per_class=5000, augment=True)
        val_set = EMNISTBalanced(split='val', samples_per_class=1000, augment=False)

        # Calculate safe number of workers
        num_cpu = os.cpu_count()
        num_workers = min(8, num_cpu - 1) if num_cpu and num_cpu > 1 else 0

        # Create data loaders with async loading
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            generator=generator  # Für Reproduzierbarkeit
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

        # Initialize model
        if args.model == 'base':
            model = BaseClassifier(
                num_classes=36,
                freeze_layers=best_params.get('freeze_layers', 8)
            )
        elif args.model == 'modular':
            base_model = BaseClassifier(
                num_classes=36,
                freeze_layers=best_params.get('freeze_layers', 8)
            )
            category_model = CategoryClassifier()
            model = ModularClassifier(
                base_model,
                category_model,
                temp_class=best_params.get('temp_class', 1.0),
                temp_cat=best_params.get('temp_cat', 1.0),
                alpha=best_params.get('alpha', 0.5)
            )
        else:  # Category-Modell
            model = CategoryClassifier()

        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=best_params['lr'],
            weight_decay=best_params.get('weight_decay', 1e-4),
            max_epochs=args.epochs
        )

        # Train model
        start_time = time.time()
        history = trainer.train(save_path=f'best_{args.model}_model.pth')
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")

        # Save training history
        with open(f'{args.model}_training_history.json', 'w') as f:
            json.dump(history, f, indent=4)

    # Model evaluation
    if args.evaluate:
        print(f"\n{'=' * 50}")
        print(f"Evaluating {args.model} model")
        print(f"{'=' * 50}")

        # Try to load best parameters for model initialization
        param_file = f'best_{args.model}_params.json'
        best_params = {}
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                best_params = json.load(f)

        # Initialize model
        if args.model == 'base':
            model = BaseClassifier(
                num_classes=36,
                freeze_layers=best_params.get('freeze_layers', 8)
            )
        elif args.model == 'modular':
            base_model = BaseClassifier(
                num_classes=36,
                freeze_layers=best_params.get('freeze_layers', 8)
            )
            category_model = CategoryClassifier()
            model = ModularClassifier(
                base_model,
                category_model,
                temp_class=best_params.get('temp_class', 1.0),
                temp_cat=best_params.get('temp_cat', 1.0),
                alpha=best_params.get('alpha', 0.5)
            )
        else:  # Category-Modell
            model = CategoryClassifier()

        # Load trained model
        model_path = f'best_{args.model}_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"Loaded trained model from {model_path}")
        else:
            print("Warning: No trained model found. Evaluating untrained model.")
            model.to(device)

        # Evaluate
        start_time = time.time()
        accuracy, cm_fig = evaluate_model(model, device, args.model)
        eval_time = time.time() - start_time

        print(f"\nEvaluation completed in {eval_time:.2f}s")
        print(f"Test accuracy: {accuracy:.4f}")

        # Save confusion matrix
        cm_path = f'{args.model}_confusion_matrix.png'
        cm_fig.savefig(cm_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")

        # Save evaluation results
        results = {
            'model_type': args.model,
            'accuracy': accuracy,
            'evaluation_time': eval_time
        }
        with open(f'{args.model}_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
