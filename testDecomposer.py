
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from ComponentDecomposer import *
from ToyPileUpGenerator import *
from ToyPileUpPlotter import *


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "./datasets/decomposition"
MODEL_DIR = "./models/decomposer"
OUTPUT_DIR = "./decomposer_test"

END_NAME = "_N2"

TEST_FILE = os.path.join(
    DATA_DIR,
    "decomposition"+END_NAME+"_test.npz"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "decomposer"+END_NAME+".pt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Number of example events to plot
N_EXAMPLES = 5

# Number of bins used for parameter-dependent performance plots
N_PARAMETER_BINS = 10

# Minimum component peak required to attempt a fit
MIN_COMPONENT_PEAK = 1e-5

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# Pulse model
# =========================================================

def pulse_function(t, amplitude, arrival_time, rise_time, fall_time):

    dt = t - arrival_time

    pulse = np.zeros_like(t, dtype=np.float64)

    mask = dt >= 0

    pulse[mask] = (
        np.exp(-dt[mask] / fall_time)
        - np.exp(-dt[mask] / rise_time)
    )

    maximum = np.max(pulse)

    if maximum > 0:
        pulse /= maximum

    return amplitude * pulse


# =========================================================
# Fit individual pulse components
# =========================================================

def fit_pulse(waveform):

    waveform = np.asarray(waveform, dtype=np.float64)

    t = np.arange(len(waveform), dtype=np.float64)

    # Ignore empty or near-empty components
    if np.max(waveform) < MIN_COMPONENT_PEAK:
        return np.full(4, np.nan)

    peak_idx = np.argmax(waveform)
    peak_amplitude = waveform[peak_idx]

    # Initial parameter estimates
    p0 = [
        peak_amplitude,
        max(-len(waveform), peak_idx - 2.0),
        1.0,
        3.0
    ]

    # Fit rise time and the positive difference (fall_time - rise_time).
    # This ensures fall_time is always greater than rise_time.
    def fit_function(t, amplitude, arrival_time, rise_time, delta_time):

        fall_time = rise_time + delta_time

        return pulse_function(
            t,
            amplitude,
            arrival_time,
            rise_time,
            fall_time
        )

    lower_bounds = [
        0.0,
        -len(waveform),
        0.05,
        0.05
    ]

    upper_bounds = [
        np.inf,
        peak_idx + 0.1,
        2.0 * len(waveform),
        2.0 * len(waveform)
    ]

    try:

        popt, _ = curve_fit(
            fit_function,
            t,
            waveform,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000
        )

        amplitude = popt[0]
        arrival_time = popt[1]
        rise_time = popt[2]
        fall_time = popt[2] + popt[3]

        return np.array([
            amplitude,
            arrival_time,
            rise_time,
            fall_time
        ])

    except (RuntimeError, ValueError, FloatingPointError):

        return np.full(4, np.nan)


# =========================================================
# Fit all components in a dataset
# =========================================================

def fit_components(components):

    """
    components shape:
        (n_events, n_components, n_bins)

    Returns:
        parameters shape:
        (n_events, n_components, 4)
    """

    n_events = components.shape[0]
    n_components = components.shape[1]

    parameters = np.full(
        (n_events, n_components, 4),
        np.nan
    )

    for event in range(n_events):

        if (event % 5000) == 0:
          print(f'\tEvent {event}/{n_events}...')

        for component in range(n_components):

            parameters[event, component] = fit_pulse(
                components[event, component]
            )

    return parameters


# =========================================================
# Load test dataset
# =========================================================

data = np.load(TEST_FILE)

X_test = data["X"].astype(np.float32)[:20000]
y_test = data["y"].astype(np.float32)[:20000]

n_bins = X_test.shape[1]
n_components = y_test.shape[1]

print("\nTest dataset:")
print("Mixed waveforms:", X_test.shape)
print("True components:", y_test.shape)


# =========================================================
# Load best model
# =========================================================

model = ComponentDecomposer.load_from_state_dict(
    MODEL_FILE,
    n_bins=n_bins,
    n_components=n_components
)

model = model.to(device)
model.eval()

print("\nLoaded model:", MODEL_FILE)


# =========================================================
# Apply model to test dataset
# =========================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

predicted_components = []

with torch.no_grad():

    for start in range(0, len(X_tensor), 256):

        batch = X_tensor[start:start + 256]

        prediction = model(batch)

        predicted_components.append(
            prediction.cpu().numpy()
        )

predicted_components = np.concatenate(
    predicted_components,
    axis=0
)

print("\nPredicted components:", predicted_components.shape)

# =========================================================
# Plot example decompositions
# =========================================================

true_plotter = ToyPileUpPlotter(
    outdir=os.path.join(OUTPUT_DIR, "examples"),
    endName=END_NAME+"_true",
    n_bins=n_bins
)

pred_plotter = ToyPileUpPlotter(
    outdir=os.path.join(OUTPUT_DIR, "examples"),
    endName=END_NAME+"_pred",
    n_bins=n_bins
)

n_examples = min(N_EXAMPLES, len(X_test))

example_indices = np.random.choice(
    len(X_test),
    size=n_examples,
    replace=False
)

for idx in example_indices:

    print("Plotting event:", idx)

    # True decomposition
    true_plotter.plot_decomposition(
        X_test[idx],
        y_test[idx],
        idx
    )

    # Predicted decomposition
    pred_plotter.plot_decomposition(
        X_test[idx],
        predicted_components[idx],
        idx
    )


# =========================================================
# Fit true and predicted components
# =========================================================

print("\nFitting true components...")

true_parameters = fit_components(y_test)

print("Fitting predicted components...")

predicted_parameters = fit_components(predicted_components)


# =========================================================
# Save fitted parameters
# =========================================================

np.savez_compressed(
    os.path.join(
        OUTPUT_DIR,
        f"fitted_parameters{END_NAME}.npz"
    ),
    true_parameters=true_parameters,
    predicted_parameters=predicted_parameters
)

# =========================================================
# Plot parameter performance
# =========================================================

plotter = ToyPileUpPlotter(
    outdir=os.path.join(OUTPUT_DIR),
    endName=END_NAME,
    n_bins=n_bins
)

print("\nParameter performance:")

plotter.plot_parameter_performance(
    true_parameters,
    predicted_parameters,
    n_par_bins=N_PARAMETER_BINS
)

print("\nSimilarity:") 

best_event, worst_event = plotter.plot_component_similarity(
    y_test,
    predicted_components,
    n_par_bins=N_PARAMETER_BINS
)

print("\nTesting complete.")
print("Results saved in:", OUTPUT_DIR)
