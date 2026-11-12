
import numpy as np
from pathlib import Path


class ToyPileUpGenerator:
    """
    Generate toy drift-chamber wire waveforms for pileup ML studies.

    Two datasets are supported:

    1. Classification:
       Input:  mixed waveform (n_bins,)
       Target: 0 = single waveform, 1 = pileup

       The number of waveforms is randomly selected between
       1 and max_n_waveforms.

    2. Decomposition:
       Input:  mixed waveform (n_bins,)
       Target: individual waveforms (N, n_bins)

       Exactly N waveforms are generated per event.

    Waveforms are generated using a difference of exponentials
    with variable amplitudes, arrival times, rise times and
    fall times.
    """

    def __init__(
        self,
        n_bins=30,
        max_n_waveforms=2,
        noise_std=0.03,
        baseline_std=0.02,
        amplitude_range=(0.5, 5.0),
        rise_time_range=(0.8, 2.5),
        fall_time_range=(3.0, 9.0),
        arrival_time_range=(-2.0, 18.0),
        seed=42
    ):

        self.n_bins = n_bins
        self.max_n_waveforms = max_n_waveforms

        self.noise_std = noise_std
        self.baseline_std = baseline_std

        self.amplitude_range = amplitude_range
        self.rise_time_range = rise_time_range
        self.fall_time_range = fall_time_range
        self.arrival_time_range = arrival_time_range

        self.rng = np.random.default_rng(seed)

        self.time = np.arange(n_bins, dtype=np.float32)

    # ---------------------------------------------------------
    # Individual waveform generation
    # ---------------------------------------------------------

    def generate_waveform(self):
        """
        Generate a single detector waveform.

        Returns
        -------
        waveform : np.ndarray
            Shape (n_bins,).

        parameters : dict
            Randomly sampled waveform parameters.
        """

        # Random waveform parameters
        amplitude = self.rng.uniform(*self.amplitude_range)

        t0 = self.rng.uniform(*self.arrival_time_range)

        tau_r = self.rng.uniform(*self.rise_time_range)
        tau_f = self.rng.uniform(*self.fall_time_range)

        # Ensure the fall time is larger than the rise time
        tau_f = max(tau_f, tau_r + 0.5)

        # Time relative to waveform arrival
        dt = self.time - t0

        waveform = np.zeros(self.n_bins, dtype=np.float32)

        # Difference of exponentials
        mask = dt >= 0

        waveform[mask] = (
            np.exp(-dt[mask] / tau_f)
            - np.exp(-dt[mask] / tau_r)
        )

        # Normalise pulse to unit maximum before applying amplitude
        peak = np.max(waveform)

        if peak > 0:
            waveform /= peak

        waveform *= amplitude

        parameters = {
            "amplitude": amplitude,
            "arrival_time": t0,
            "rise_time": tau_r,
            "fall_time": tau_f
        }

        return waveform, parameters

    # ---------------------------------------------------------
    # Common event generation
    # ---------------------------------------------------------

    def generate_event(self, n_waveforms):
        """
        Generate an event containing n_waveforms individual pulses.

        Returns
        -------
        mixed : np.ndarray
            Sum of individual waveforms, including noise.

        components : np.ndarray
            Individual waveforms, shape (N, n_bins).

        parameters : list
            Parameters for each individual waveform.
        """

        components = []
        parameters = []

        for _ in range(n_waveforms):

            waveform, params = self.generate_waveform()

            components.append(waveform)
            parameters.append(params)

        components = np.asarray(components, dtype=np.float32)

        # Sort individual waveforms by arrival time
        order = np.argsort(
            [p["arrival_time"] for p in parameters]
        )

        components = components[order]
        parameters = [parameters[i] for i in order]

        # Sum individual waveforms
        mixed = np.sum(components, axis=0)

        # Add event-level baseline offset
        baseline = self.rng.normal(0, self.baseline_std)

        # Add electronic noise
        noise = self.rng.normal(
            0,
            self.noise_std,
            size=self.n_bins
        )

        mixed = mixed + baseline + noise

        return mixed.astype(np.float32), components, parameters

    # ---------------------------------------------------------
    # Dataset 1: Pileup classification
    # ---------------------------------------------------------

    def generate_classification_dataset(
        self,
        n_samples=100000,
        output_dir="toy_pileup_data",
        test_size=0.2
    ):
        """
        Generate a binary pileup classification dataset.

        X: (n_samples, n_bins)
        y: (n_samples,)

        y = 0: single waveform
        y = 1: multiple waveforms
        """

        X = np.zeros(
            (n_samples, self.n_bins),
            dtype=np.float32
        )

        y = np.zeros(n_samples, dtype=np.int64)

        n_waves = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):

            # Random number of waveforms between 1 and N
            n = self.rng.integers(
                1,
                self.max_n_waveforms + 1
            )

            mixed, _, _ = self.generate_event(n)

            X[i] = mixed
            y[i] = int(n > 1)
            n_waves[i] = n

        return self._split_and_save(
            X=X,
            y=y,
            n_waves=n_waves,
            output_dir=output_dir,
            prefix="classification",
            test_size=test_size
        )

    # ---------------------------------------------------------
    # Dataset 2: Waveform decomposition
    # ---------------------------------------------------------

    def generate_decomposition_dataset(
        self,
        n_samples=10000,
        n_waveforms=None,
        output_dir="toy_pileup_data",
        test_size=0.2
    ):
        """
        Generate a waveform decomposition dataset.

        X: (n_samples, n_bins)
        y: (n_samples, N, n_bins)

        X contains the sum of N waveforms.
        y contains the individual waveforms.
        """

        if n_waveforms is None:
            n_waveforms = self.max_n_waveforms

        if n_waveforms < 1:
            raise ValueError("n_waveforms must be >= 1")

        X = np.zeros(
            (n_samples, self.n_bins),
            dtype=np.float32
        )

        y = np.zeros(
            (n_samples, n_waveforms, self.n_bins),
            dtype=np.float32
        )

        for i in range(n_samples):

            mixed, components, _ = self.generate_event(
                n_waveforms
            )

            X[i] = mixed
            y[i] = components

        n_waves = np.full(
            n_samples,
            n_waveforms,
            dtype=np.int64
        )

        return self._split_and_save(
            X=X,
            y=y,
            n_waves=n_waves,
            output_dir=output_dir,
            prefix=f"decomposition_N{n_waveforms}",
            test_size=test_size
        )

    # ---------------------------------------------------------
    # Train/test splitting and saving
    # ---------------------------------------------------------

    def _split_and_save(
        self,
        X,
        y,
        n_waves,
        output_dir,
        prefix,
        test_size
    ):
        """
        Randomly split and save the dataset as compressed NPZ files.
        """

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle dataset indices
        indices = self.rng.permutation(len(X))

        n_test = int(len(X) * test_size)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        datasets = {}

        for name, idx in [
            ("train", train_idx),
            ("test", test_idx)
        ]:

            dataset = {
                "X": X[idx],
                "y": y[idx],
                "n_waves": n_waves[idx]
            }

            filename = output_dir / f"{prefix}_{name}.npz"

            np.savez_compressed(
                filename,
                **dataset
            )

            datasets[name] = dataset

            print(
                f"Saved {filename} "
                f"| X: {dataset['X'].shape} "
                f"| y: {dataset['y'].shape}"
            )

        return datasets
