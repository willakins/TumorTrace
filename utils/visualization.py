import matplotlib.pyplot as plt
import numpy as np
import torch

class Visualizer:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    # Function to plot loss curve
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.show()

    # Function to plot accuracy curve
    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.show()

    # Function to save the loss and accuracy plots
    def save_plots(self, path="/results/figures/"):
        import os
        os.makedirs(path, exist_ok=True)

        self.plot_loss()
        plt.savefig(os.path.join(path, 'loss_plot.png'))

        self.plot_accuracy()
        plt.savefig(os.path.join(path, 'accuracy_plot.png'))

    # Function to update metrics after each epoch
    def update_metrics(self, train_loss, train_accuracy, val_accuracy):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)