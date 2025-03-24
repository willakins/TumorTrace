import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report

class Visualizer:
    def __init__(self, class_names=None):
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.class_names = class_names if class_names else ['Glioma', 'Meningioma', 'Pituitary']

    # Function to plot loss curve
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, marker='o', linestyle='-', label='Train Loss', color='tab:red')
        plt.title('Loss over Epochs', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Function to plot accuracy curves
    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, marker='o', linestyle='-', label='Train Accuracy', color='tab:blue')
        plt.plot(self.val_accuracies, marker='x', linestyle='--', label='Validation Accuracy', color='tab:green')
        plt.title('Accuracy over Epochs', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Update metrics for plotting
    def update_metrics(self, train_loss, train_accuracy, val_accuracy):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

    # Plot confusion matrix
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, title='Confusion Matrix', cmap='Blues'):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    # Print classification report
    def print_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print("\nClassification Report:\n")
        print(report)

    # Combined visualization after evaluation
    def evaluation_summary(self, y_true, y_pred):
        self.plot_confusion_matrix(y_true, y_pred)
        self.print_classification_report(y_true, y_pred)

    # Save plots to file
    def save_plots(self, path="results/figures/"):
        os.makedirs(path, exist_ok=True)

        # Save loss
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss', color='tab:red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'loss_plot.png'))

        # Save accuracy
        plt.figure()
        plt.plot(self.train_accuracies, label='Train Accuracy', color='tab:blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='tab:green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(path, 'accuracy_plot.png'))

        print(f"Saved training plots to {path}")