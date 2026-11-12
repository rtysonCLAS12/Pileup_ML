
from ToyPileUpGenerator import ToyPileUpGenerator
from ToyPileUpPlotter import ToyPileUpPlotter
import numpy as np

# Initialise generator
generator = ToyPileUpGenerator(
    n_bins=30,
    max_n_waveforms=2,
    noise_std=0.03,
    seed=42
)

# Dataset 1: Classification
classification = generator.generate_classification_dataset(
    n_samples=1000000,
    output_dir="datasets/classification",
    test_size=0.2
)

# Dataset 2: Decomposition
decomposition = generator.generate_decomposition_dataset(
    n_samples=1000000,
    n_waveforms=2,
    output_dir="datasets/decomposition",
    test_size=0.2
)

# Access generated data
X_train = classification["train"]["X"]
y_train = classification["train"]["y"]

X_train_decomp = decomposition["train"]["X"]
y_train_decomp = decomposition["train"]["y"]

print("Classification:", X_train.shape, y_train.shape)
print("Decomposition:", X_train_decomp.shape, y_train_decomp.shape)


plotter = ToyPileUpPlotter(
    outdir="./plot_examples",
    endName="_toy",
    n_bins=30
)

# Plot individual classification events
plotter.plot_classification_dataset(
    classification["train"],
    indices=range(5)
)

# Plot individual decomposition events
plotter.plot_decomposition_dataset(
    decomposition["train"],
    indices=range(5)
)
