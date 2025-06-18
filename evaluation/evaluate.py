import os
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.dataset import EMNISTBalanced
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(model, data_loader, device, class_names):
    """Plot confusion matrix with classification report"""
    model.eval()
    all_preds = []
    all_labels = []

    # Evaluate with progress bar
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(torch.exp(outputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Generate classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save classification report
    with open('classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    return plt.gcf(), report


def evaluate_model(model, device, model_type='base'):
    """Comprehensive model evaluation"""
    # Create test dataset
    test_set = EMNISTBalanced(split='test', samples_per_class=1000, augment=False)
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        num_workers=min(8, os.cpu_count() - 1),
        pin_memory=True
    )

    # Generate class names - CORRECTED VERSION
    class_names = (
            [str(i) for i in range(10)] +
            [chr(ord('A') + i) for i in range(13)] +
            [chr(ord('a') + i) for i in range(13)]
    )

    # Evaluate
    start_time = time.time()
    cm_fig, report = plot_confusion_matrix(model, test_loader, device, class_names)
    accuracy = report['accuracy']
    eval_time = time.time() - start_time

    # Print top-level results
    print("\nClassification Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds")

    # Print per-class accuracy for digits, uppercase, lowercase
    categories = {
        'digits': class_names[:10],
        'uppercase': class_names[10:23],
        'lowercase': class_names[23:36]
    }

    for cat_name, classes in categories.items():
        cat_acc = np.mean([report[cls]['recall'] for cls in classes])
        print(f"{cat_name.capitalize()} Accuracy: {cat_acc:.4f}")

    return accuracy, cm_fig