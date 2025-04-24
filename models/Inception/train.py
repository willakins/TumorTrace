import torch
from sklearn.metrics import accuracy_score
from utils.visualization import Visualizer
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None, aux_loss_weight=0.4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aux_loss_weight = aux_loss_weight
        
        self.model.to(self.device)

        self.visualizer = Visualizer()


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
                #self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                # Check whether the model includes auxiliary outputs
                if isinstance(outputs, tuple) and self.model.training:
                    main_out, aux_out = outputs
                    loss_main = self.criterion(main_out, labels)
                    loss_aux = self.criterion(aux_out, labels)
                    loss = loss_main + self.aux_loss_weight * loss_aux
                    
                    outputs_for_acc = main_out
                else:
                    loss = self.criterion(outputs, labels)
                    outputs_for_acc = outputs

                # Compute loss
                #loss = self.criterion(outputs, labels)
                print(f'Batch number {batch_idx}, Loss: {loss}')
                running_loss += loss.item()

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs_for_acc, 1) # Use main output for accuracy
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct_predictions / total_samples

            # Validation step
            val_accuracy = self.evaluate()
            
            # Update visualizer metrics
            self.visualizer.update_metrics(epoch_loss, epoch_accuracy, val_accuracy)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy*100:.2f}%, Validation Accuracy: {val_accuracy*100:.2f}%')

            # Save the model checkpoint if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()

        # After training, plot the metrics
        self.visualizer.plot_loss()
        self.visualizer.plot_accuracy()

    def save_model(self, save_path="models/Inception/checkpoints/inception.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    def evaluate(self, return_preds=False):
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

        if return_preds:
            return accuracy, all_labels, all_preds
        else:
            return accuracy