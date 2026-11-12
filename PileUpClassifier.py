import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

import numpy as np

class ClassificationDataset(Dataset):

    def __init__(self, filename):

        data = np.load(filename)

        self.X = torch.tensor(
            data["X"],
            dtype=torch.float32
        )

        self.y = torch.tensor(
            data["y"],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return {
            "mixed": self.X[idx],
            "label": self.y[idx]
        }

class PileUpClassifier(pl.LightningModule):
    """
    Predicts whether a mixed histogram contains pileup.

    Input:
        mixed: (B, n_bins)

    Output:
        logits: (B, 1)

    Target:
        0 = no pileup
        1 = pileup
    """

    __annotations__ = {}

    def __init__(
        self,
        n_bins,
        lr=1e-3,
        threshold=0.5
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_bins = n_bins
        self.lr = lr
        self.threshold = threshold

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * n_bins, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Mixed waveform, shape (B, n_bins).

        Returns
        -------
        torch.Tensor
            Pileup logits, shape (B, 1).
        """

        x = x.unsqueeze(1)

        logits = self.net(x)

        return logits

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict_proba(self, x):
        """
        Returns probability of pileup, shape (B,).
        """

        logits = self.forward(x)

        return torch.sigmoid(logits).squeeze(1)

    def predict(self, x):
        """
        Returns binary pileup prediction, shape (B,).
        """

        probabilities = self.predict_proba(x)

        return (probabilities >= self.threshold).long()

    # ---------------------------------------------------------
    # Loss and metrics
    # ---------------------------------------------------------

    def calculate_metrics(self, logits, target):

        loss = self.loss_fn(
            logits.squeeze(1),
            target.float()
        )

        probabilities = torch.sigmoid(logits.squeeze(1))

        predictions = (
            probabilities >= self.threshold
        ).float()

        accuracy = (predictions == target.float()).float().mean()

        return loss, accuracy

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def training_step(self, batch, batch_idx):

        x = batch["mixed"]
        y = batch["label"]

        logits = self.forward(x)

        loss, accuracy = self.calculate_metrics(
            logits,
            y
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True,  prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validation_step(self, batch, batch_idx):

        x = batch["mixed"]
        y = batch["label"]

        logits = self.forward(x)

        loss, accuracy = self.calculate_metrics(
            logits,
            y
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ---------------------------------------------------------
    # Optimiser
    # ---------------------------------------------------------

    def configure_optimizers(self):

        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    def export_to_state_dict(self, path: str):

        self.to("cpu").eval()

        torch.save(self.state_dict(), path)

        print(f"Saved state_dict to {path}")

    def export_to_torchscript(self, path: str):

        self.to("cpu").eval()

        scripted = torch.jit.script(self.net)

        scripted.save(path)

        print(f"TorchScript model saved to {path}")

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    @staticmethod
    def load_from_state_dict(
        path: str,
        n_bins: int,
        lr=1e-3,
        threshold=0.5
    ):

        model = PileUpClassifier(
            n_bins=n_bins,
            lr=lr,
            threshold=threshold
        )

        sd = torch.load(path, map_location="cpu")

        model.load_state_dict(sd)

        model.eval()

        return model

    @staticmethod
    def load_from_torchscript(path: str):

        model = torch.jit.load(path, map_location="cpu")

        model.eval()

        return model

class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())