import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from utils.utils import compute_accuracy, compute_loss
from data.image_loader import ImageLoader
from src.models import (
    CNN_3D,
    MyResNet,
    MyInception,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import seaborn as sns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
        self,
        data_dir: str,
        model: Union[CNN_3D, MyResNet, MyInception],
        optimizer: Optimizer,
        model_dir: str,
        train_data_transforms: transforms.Compose,
        val_data_transforms: transforms.Compose,
        batch_size: int = 100,
        load_from_disk: bool = True,
        cuda: bool = False,
        n_slices = 1,
    ) -> None:
        self.model_dir = model_dir

        self.model = model

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

        is_3d = model.__class__.__name__ == "CNN_3D"

        self.train_dataset = ImageLoader(
            data_dir, split="train", transform=train_data_transforms, is_3d=is_3d, n_slices=n_slices
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.val_dataset = ImageLoader(
            data_dir, split="test", transform=val_data_transforms, is_3d=is_3d, n_slices=n_slices
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

    def save_model(self) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, "checkpoint.pt"),
        )

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_loss, train_acc = self.train_epoch()

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)

            val_loss, val_acc = self.validate()
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy: {train_acc:.4f}"
                + f" Validation Accuracy: {val_acc:.4f}"
            )
    
    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # loop over each minibatch
        for batch_idx, (x, y) in enumerate(self.train_loader):
            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            # for 3D-CNN: data comes in as [N, n_slices, 1, H, W]
            # we need [N, in_channels=1, depth=n_slices, H, W]
            if isinstance(self.model, CNN_3D):
                x = x.permute(0, 2, 1, 3, 4)

            n = x.shape[0]
            logits = self.model(x)

            # Necessary for inception
            if isinstance(logits, tuple):
                logits = logits[0]

            batch_acc = compute_accuracy(logits, y)
            train_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            print(
                f"Minibatch:{batch_idx + 1}"
                + f" Train Loss:{batch_loss:.4f}"
                + f" Val Loss: {batch_loss:.4f}"
                + f" Train Accuracy: {batch_acc:.4f}"
                + f" Validation Accuracy: {batch_acc:.4f}"
            )

        return train_loss_meter.avg, train_acc_meter.avg

    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter("val loss")
        val_acc_meter = AverageMeter("val accuracy")

        with torch.no_grad():
            # loop over whole val set
            for (x, y) in self.val_loader:
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                if isinstance(self.model, CNN_3D):
                    x = x.permute(0, 2, 1, 3, 4)

                n = x.shape[0]
                logits = self.model(x)

                batch_acc = compute_accuracy(logits, y)
                val_acc_meter.update(val=batch_acc, n=n)

                batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
                val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        self.model.train()

        return val_loss_meter.avg, val_acc_meter.avg

    def plot_loss_history(self) -> None:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(len(self.train_loss_history)), y=self.train_loss_history, label="Training", color="tab:red", marker='o')
        sns.lineplot(x=range(len(self.validation_loss_history)), y=self.validation_loss_history, label="Validation", color="tab:purple", marker='x')
        plt.title(f"{self.model.__class__.__name__} Loss History", fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self) -> None:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(len(self.train_accuracy_history)), y=self.train_accuracy_history, label="Training", color="tab:blue", marker='o')
        sns.lineplot(x=range(len(self.validation_accuracy_history)), y=self.validation_accuracy_history, label="Validation", color="tab:green", marker='x')
        plt.title(f"{self.model.__class__.__name__} Accuracy History", fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    from sklearn.metrics import classification_report, f1_score

    def print_classification_report(self) -> float:
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.val_loader:
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                
        # Filter to only known class indices if needed
        valid_classes = [0, 1, 2]
        report = classification_report(
            all_labels, all_preds,
            labels=valid_classes,
            target_names=['Glioma', 'Meningioma', 'Pituitary'],
            zero_division=0
        )
        macro_f1 = f1_score(all_labels, all_preds, labels=valid_classes, average='macro', zero_division=0)

        print("\nClassification Report:\n")
        print(report)
        print(f"Macro F1 Score: {macro_f1:.4f}")

    def save_plots(self, path):
        os.makedirs(path, exist_ok=True)

        # Save loss
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))

        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(path, f'{self.model.__class__.__name__}_loss_plot.png'))
        plt.close()

        # Save accuracy
        plt.figure()
        epoch_idxs = range(len(self.train_accuracy_history))
        plt.plot(epoch_idxs, self.train_accuracy_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_accuracy_history, "-r", label="validation")
        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(path, f'{self.model.__class__.__name__}_accuracy_plot.png'))
        plt.close()

        print(f"Saved training plots to {path}")
