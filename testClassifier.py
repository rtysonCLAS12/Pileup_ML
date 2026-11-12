
import os
import numpy as np
import torch

from PileUpClassifier import *
from ToyPileUpPlotter import *


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "./datasets/classification"
DATA_DECOMP_DIR = "./datasets/decomposition"
MODEL_DIR = "./models/classifier"
OUTPUT_DIR = "./classifier_test"

END_NAME = ""

TEST_FILE = os.path.join(
    DATA_DIR,
    "classification"+END_NAME+"_test.npz"
)

TEST_DECOMP_FILE = os.path.join(
    DATA_DECOMP_DIR,
    "decomposition_N2_test.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "classifier"+END_NAME+".pt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# Device configuration
# =========================================================

if torch.cuda.is_available():

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    print("Using device:", torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():

    device = torch.device("mps")
    print("Using MPS")

else:

    device = torch.device("cpu")
    print("Using CPU")


# =========================================================
# Load test dataset
# =========================================================

data = np.load(TEST_FILE)

X_test = data["X"].astype(np.float32)
y_test = data["y"].astype(np.int64)

data_decomp = np.load(TEST_DECOMP_FILE)

X_test_decomp = data_decomp["X"].astype(np.float32)
y_test_decomp = data_decomp["y"].astype(np.float32)

n_bins = X_test.shape[1]

print("\nTest dataset:")
print("Number of events:", len(X_test))
print("Number of bins:", n_bins)
print("Pileup events:", np.sum(y_test == 1))
print("No-pileup events:", np.sum(y_test == 0))


# =========================================================
# Load best model
# =========================================================

model = PileUpClassifier.load_from_state_dict(
    MODEL_FILE,
    n_bins=n_bins
)

model = model.to(device)
model.eval()

print("\nLoaded model:", MODEL_FILE)


# =========================================================
# Apply classifier to test dataset
# =========================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

predicted_probabilities = []

with torch.no_grad():

    for start in range(0, len(X_tensor), 256):

        batch = X_tensor[start:start + 256]

        probabilities = model.predict_proba(batch)

        predicted_probabilities.append(
            probabilities.cpu().numpy()
        )

predicted_probabilities = np.concatenate(
    predicted_probabilities,
    axis=0
).flatten()


# =========================================================
# Save predictions
# =========================================================

np.savez_compressed(
    os.path.join(
        OUTPUT_DIR,
        f"classification_predictions{END_NAME}.npz"
    ),
    probabilities=predicted_probabilities,
    labels=y_test
)


# =========================================================
# Plot results
# =========================================================

plotter = ToyPileUpPlotter(
    outdir=OUTPUT_DIR,
    endName=END_NAME,
    n_bins=n_bins
)

plotter.plot_classification_probability(
    predicted_probabilities,
    y_test
)

best_pileup_efficiency, best_no_pileup_efficiency, best_threshold = (
    plotter.plot_classification_efficiency(
        predicted_probabilities,
        y_test,
        no_pileup_efficiency_threshold=0.95
    )
)

predicted_probabilities = []

with torch.no_grad():

    for start in range(0, len(X_test_decomp), 1024):

        xb = torch.as_tensor(
            X_test_decomp[start:start + 1024],
            dtype=torch.float32,
            device=device
        )

        prob = model.predict_proba(xb)

        predicted_probabilities.append(
            prob.detach().cpu().numpy()
        )

predicted_probabilities = np.concatenate(predicted_probabilities)
predicted_pileup = predicted_probabilities >= best_threshold

# Evaluate decomposition
results = plotter.evaluate_pileup_decomposition(
    components=y_test_decomp,
    predicted_pileup=predicted_pileup,
    probabilities=predicted_probabilities,
    threshold=best_threshold
)


print("\nTesting complete.")
print("Results saved in:", OUTPUT_DIR)
