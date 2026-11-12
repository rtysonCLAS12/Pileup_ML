import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

import numpy as np


# =========================================================
# Dataset
# =========================================================

class DecompositionDataset(Dataset):

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

        print(
            f"Loaded {filename}: "
            f"X = {self.X.shape}, "
            f"y = {self.y.shape}"
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return {
            "mixed": self.X[idx],
            "components": self.y[idx]
        }


class ComponentDecomposer(pl.LightningModule):
    """
    Predicts individual component histograms from a mixed histogram.

    Input:
        mixed:      (B, n_bins)

    Output:
        components: (B, n_components, n_bins)

    Enforces sum constraint via loss.
    """

    __annotations__ = {}

    def __init__(
        self,
        n_bins,
        n_components,
        lr=1e-3,
        lambda_nevs=1.0
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_bins = n_bins
        self.n_components = n_components
        self.lr = lr
        self.lambda_nevs = lambda_nevs

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, n_components, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_components),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                n_components * n_bins,
                n_components * n_bins
            ),
            nn.ReLU()
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
            Predicted components, shape (B, n_components, n_bins).
        """

        x = x.unsqueeze(1)

        x = self.net(x)

        x = x.view(
            -1,
            self.n_components,
            self.n_bins
        )

        return x

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def loss_fn(self, pred, target, x_mixed):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted components, (B, C, n_bins).

        target : torch.Tensor
            True components, (B, C, n_bins).

        x_mixed : torch.Tensor
            Mixed waveform, (B, n_bins).
        """

        # Individual component reconstruction
        loss_comp = F.mse_loss(pred, target)

        # Sum constraint
        pred_nevs = pred.sum(dim=1)

        loss_nevs = F.mse_loss(pred_nevs, x_mixed)

        loss = loss_comp + self.lambda_nevs * loss_nevs

        return loss, loss_comp, loss_nevs

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def training_step(self, batch, batch_idx):

        x = batch["mixed"]
        y = batch["components"]

        pred = self.forward(x)

        loss, loss_comp, loss_nevs = self.loss_fn(
            pred,
            y,
            x
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_comp", loss_comp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_nevs", loss_nevs, on_step=False, on_epoch=True, prog_bar=False)

        # return loss # sum of loss to predict components and make sure they sum to mixed hist total
        
        return loss_comp # only loss to predict components

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validation_step(self, batch, batch_idx):

        x = batch["mixed"]
        y = batch["components"]

        pred = self.forward(x)

        loss, loss_comp, loss_nevs = self.loss_fn(
            pred,
            y,
            x
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_comp", loss_comp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_nevs", loss_nevs, on_step=False, on_epoch=True, prog_bar=False)

        # return loss # sum of loss to predict components and make sure they sum to mixed hist total
        return loss_comp # only loss to predict components

    # ---------------------------------------------------------
    # Optimiser
    # ---------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def configure_optimizers(self):

    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.lr
    #     )

    #     flat_epochs = 200
    #     cosine_epochs = 200

    #     scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimizer,
    #         schedulers=[
    #             torch.optim.lr_scheduler.ConstantLR(
    #                 optimizer,
    #                 factor=1.0,
    #                 total_iters=flat_epochs
    #             ),

    #             torch.optim.lr_scheduler.CosineAnnealingLR(
    #                 optimizer,
    #                 T_max=cosine_epochs,
    #                 eta_min=1e-5
    #             )
    #         ],
    #         milestones=[flat_epochs]
    #     )

    #     return [optimizer], [scheduler]

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
        n_components: int,
        lr=1e-3,
        lambda_nevs=1.0
    ):

        model = ComponentDecomposer(
            n_bins=n_bins,
            n_components=n_components,
            lr=lr,
            lambda_nevs=lambda_nevs
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
        self.train_losses_comp = []
        self.val_losses_comp = []
        self.train_losses_nevs = []
        self.val_losses_nevs = []

    def on_train_epoch_end(self, trainer, pl_module):

        self.train_losses.append(
            trainer.callback_metrics["train_loss"].item()
        )
        self.train_losses_comp.append(
            trainer.callback_metrics["train_loss_comp"].item()
        )
        self.train_losses_nevs.append(
            trainer.callback_metrics["train_loss_nevs"].item()
        )

    def on_validation_epoch_end(self, trainer, pl_module):

        self.val_losses.append(
            trainer.callback_metrics["val_loss"].item()
        )
        self.val_losses_comp.append(
            trainer.callback_metrics["val_loss_comp"].item()
        )
        self.val_losses_nevs.append(
            trainer.callback_metrics["val_loss_nevs"].item()
        )

