
import os
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from PileUpClassifier import *
from ToyPileUpPlotter import ToyPileUpPlotter


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "./datasets/classification"

END_NAME = ""

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "classification"+END_NAME+"_train.npz"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "classification"+END_NAME+"_test.npz"
)

OUTPUT_DIR = "./models/classifier"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Training parameters
BATCH_SIZE = 64
MAX_EPOCHS = 500
LEARNING_RATE = 1e-3

# Reproducibility
SEED = 42

pl.seed_everything(SEED, workers=True)


# =========================================================
# Device configuration
# =========================================================

if torch.cuda.is_available():

    accelerator = "gpu"
    devices = 1

    # CUDA helping things
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.set_float32_matmul_precision("high")

    torch.cuda.init()
    torch.cuda.empty_cache()

    print("Using device:", torch.cuda.get_device_name(0))

    numworkers = 4

elif torch.backends.mps.is_available():

    devices = 1
    accelerator = "mps"

    numworkers = 0

    torch.set_float32_matmul_precision("medium")

else:

    print("GPU requested but not available. Falling back to CPU.")

    accelerator = "cpu"
    devices = 1
    numworkers = 4


# =========================================================
# Load datasets
# =========================================================

train_dataset = ClassificationDataset(TRAIN_FILE)
test_dataset = ClassificationDataset(TEST_FILE)

n_bins = train_dataset.X.shape[1]

print("\nDataset configuration:")
print("Number of bins:", n_bins)
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

print(
    "Training pileup fraction:",
    train_dataset.y.mean().item()
)

print(
    "Testing pileup fraction:",
    test_dataset.y.mean().item()
)


# =========================================================
# DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=numworkers,
    pin_memory=(accelerator == "gpu"),
    persistent_workers=(numworkers > 0)
)

val_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=numworkers,
    pin_memory=(accelerator == "gpu"),
    persistent_workers=(numworkers > 0)
)


# =========================================================
# Model
# =========================================================

model = PileUpClassifier(
    n_bins=n_bins,
    lr=LEARNING_RATE
)


# =========================================================
# Callbacks
# =========================================================

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename="classifier-{epoch:03d}-{val_loss:.6f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)

loss_tracker = LossTracker()


# =========================================================
# Trainer
# =========================================================

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator=accelerator,
    devices=devices,
    callbacks=[
        checkpoint_callback,
        early_stop_callback,
        loss_tracker
    ],
    log_every_n_steps=10,
    deterministic=True
)


# =========================================================
# Training
# =========================================================

print("\nStarting training...\n")

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)


# =========================================================
# Save best model
# =========================================================

best_path = os.path.join(
    OUTPUT_DIR,
    "classifier"+END_NAME+".pt"
)

# Restore the weights from the best validation checkpoint
best_model = PileUpClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    n_bins=n_bins,
    lr=LEARNING_RATE
)

# Export the best model's state dict
best_model.export_to_state_dict(best_path)

print("\nBest checkpoint:", checkpoint_callback.best_model_path)
print("Best model saved to:", best_path)


# =========================================================
# Plot training loss
# =========================================================

plotter = ToyPileUpPlotter(
    outdir="./plot_losses",
    endName=END_NAME,
    n_bins=30
)

plotter.plotTrainLoss(loss_tracker)

print("\nTraining complete.")
