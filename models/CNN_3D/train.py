import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)


    def train(self):
        best_val_accuracy = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Training loop
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct_predictions / total_samples

            # Validation step
            val_accuracy = self.evaluate()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy*100:.2f}%, Validation Accuracy: {val_accuracy*100:.2f}%')

            # Save the model checkpoint if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def save_model(self, save_path="models/3D_CNN/checkpoints/cnn_3d.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")