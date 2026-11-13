
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 40,
    'font.serif': 'Times New Roman',
    'legend.edgecolor': 'white',
    'legend.fontsize': 40,
    'lines.linewidth': 2,
    'errorbar.capsize': 5,
    'lines.marker': "s",
    'lines.markersize': 12,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'patch.linewidth': 3.0,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.major.width': 3,
    'xtick.major.width': 3,
    'axes.linewidth': 3
})


class ToyPileUpPlotter:

    def __init__(
        self,
        outdir="./plots",
        endName="",
        n_bins=30
    ):

        self.outdir = outdir
        self.endName = endName
        self.n_bins = n_bins

        os.makedirs(self.outdir, exist_ok=True)

        self.edges = np.arange(self.n_bins + 1)

        self.colors = plt.cm.tab10.colors

    # ---------------------------------------------------------
    # Classification plotting
    # ---------------------------------------------------------

    def plot_classification(
        self,
        x,
        y,
        idx,
        logscale=False
    ):
        """
        Plot a mixed waveform and its pileup classification.

        Parameters
        ----------
        x : np.ndarray
            Mixed waveform, shape (n_bins,).

        y : int
            Classification label:
            0 = no pileup
            1 = pileup

        idx : int
            Event index.
        """

        plt.figure(figsize=(20, 20))

        # Mixed waveform
        plt.hist(
            self.edges[:-1],
            bins=self.edges,
            weights=x,
            histtype="step",
            color="black",
            linewidth=5,
            label="Mixed waveform"
        )

        # Classification label
        if y == 1:
            title = "Is pileup"
        else:
            title = "Is not pileup"

        if logscale:
            plt.yscale("log")

        plt.xlabel("Time bin")
        plt.ylabel("Amplitude")

        plt.title(f"Sample {idx}: {title}")

        plt.legend()
        plt.tight_layout()

        filename = os.path.join(
            self.outdir,
            f"sample_{idx}_classification{self.endName}.png"
        )

        plt.savefig(filename, dpi=150)
        plt.close()

    # ---------------------------------------------------------
    # Decomposition plotting
    # ---------------------------------------------------------

    def plot_decomposition(
        self,
        x,
        y,
        idx,
        logscale=False,
        with_legend=True
    ):
        """
        Plot a mixed waveform and its individual components.

        Parameters
        ----------
        x : np.ndarray
            Mixed waveform, shape (n_bins,).

        y : np.ndarray
            Individual waveforms, shape (N, n_bins).

        idx : int
            Event index.
        """

        plt.figure(figsize=(20, 20))

        n_components = len(y)

        # Mixed waveform
        plt.hist(
            self.edges[:-1],
            bins=self.edges,
            weights=x,
            histtype="step",
            color="black",
            linewidth=5,
            label="Mixed waveform"
        )

        # Individual waveforms
        for i in range(n_components):

            component = y[i]

            label = f"Component {i}"

            plt.hist(
                self.edges[:-1],
                bins=self.edges,
                weights=component,
                histtype="step",
                linestyle="--",
                color=self.colors[i % len(self.colors)],
                linewidth=5,
                label=label
            )

        if logscale:
            plt.yscale("log")

        plt.xlabel("Time bin")
        plt.ylabel("Amplitude")

        plt.title(f"Sample {idx}: Waveform decomposition")

        if with_legend:
            plt.legend()

        plt.tight_layout()

        filename = os.path.join(
            self.outdir,
            f"sample_{idx}_decomposition{self.endName}.png"
        )

        plt.savefig(filename, dpi=150)
        plt.close()

    # ---------------------------------------------------------
    # Plot datasets
    # ---------------------------------------------------------

    def plot_classification_dataset(
        self,
        dataset,
        indices=None
    ):
        """
        Plot selected events from a classification dataset.
        """

        X = dataset["X"]
        y = dataset["y"]

        if indices is None:
            indices = range(len(X))

        for idx in indices:

            self.plot_classification(
                x=X[idx],
                y=y[idx],
                idx=idx
            )

    def plot_decomposition_dataset(
        self,
        dataset,
        indices=None
    ):
        """
        Plot selected events from a decomposition dataset.
        """

        X = dataset["X"]
        y = dataset["y"]

        if indices is None:
            indices = range(len(X))

        for idx in indices:

            self.plot_decomposition(
                x=X[idx],
                y=y[idx],
                idx=idx
            )

    def plotTrainLoss(self,tracker):
        train_losses=tracker.train_losses
        val_losses=tracker.val_losses

        plt.figure(figsize=(20,20))
        plt.plot(train_losses, label='Train',color='royalblue')
        plt.plot(val_losses, label='Test',color='firebrick') 
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.outdir}/loss{self.endName}.png"
        plt.savefig(outname)
        plt.close()

    def plotTrainLossDecomp(self,tracker):
        train_losses=tracker.train_losses
        val_losses=tracker.val_losses
        train_losses_comp=tracker.train_losses_comp
        val_losses_comp=tracker.val_losses_comp
        train_losses_nevs=tracker.train_losses_nevs
        val_losses_nevs=tracker.val_losses_nevs

        plt.figure(figsize=(20,20))
        plt.plot(train_losses, label='Train',color='royalblue')
        plt.plot(val_losses, label='Test',color='firebrick') 
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.outdir}/loss_decomp{self.endName}.png"
        plt.savefig(outname)
        plt.close()

        plt.figure(figsize=(20,20))
        plt.plot(train_losses_comp, label='Train',color='royalblue')
        plt.plot(val_losses_comp, label='Test',color='firebrick') 
        plt.xlabel("Epoch")
        plt.ylabel("Component Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.outdir}/loss_comp{self.endName}.png"
        plt.savefig(outname)
        plt.close()

        plt.figure(figsize=(20,20))
        plt.plot(train_losses_nevs, label='Train',color='royalblue')
        plt.plot(val_losses_nevs, label='Test',color='firebrick') 
        plt.xlabel("Epoch")
        plt.ylabel("# Events Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.outdir}/loss_nevs{self.endName}.png"
        plt.savefig(outname)
        plt.close()


    # =========================================================
    # Plot classifier output, efficiency, purity
    # =========================================================
    
    def plot_classification_probability(
        self,
        probabilities,
        labels
    ):

        fig, ax = plt.subplots(figsize=(20, 20))

        no_pileup = probabilities[labels == 0]
        pileup = probabilities[labels == 1]

        bins = np.linspace(0, 1, 101)

        ax.hist(
            no_pileup,
            bins=bins,
            histtype="stepfilled",
            linewidth=2,
            alpha=0.8,
            label="No pileup",
            color="royalblue"
        )

        ax.hist(
            pileup,
            bins=bins,
            histtype="stepfilled",
            linewidth=2,
            alpha=0.8,
            label="Pileup",
            color="firebrick"
        )

        ax.set_xlabel("Predicted pileup probability")
        ax.set_ylabel("Events")
        plt.yscale('log')
        ax.set_title("Classifier output probability")
        ax.set_xlim(0, 1)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.outdir,
                f"classification_probability{self.endName}.png"
            ),
            dpi=150
        )

        plt.close(fig)


    
    def plot_classification_efficiency(
        self,
        probabilities,
        labels,
        no_pileup_efficiency_threshold=0.95,
        thresholds=None
    ):

        if thresholds is None:
            thresholds = np.linspace(0, 1, 1001)

        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels)

        true_pileup = labels == 1
        true_no_pileup = labels == 0

        n_pileup = np.sum(true_pileup)
        n_no_pileup = np.sum(true_no_pileup)

        if n_pileup == 0 or n_no_pileup == 0:
            raise ValueError(
                "Both pileup and no-pileup events must be present."
            )

        pileup_efficiencies = []
        no_pileup_efficiencies = []

        for threshold in thresholds:

            predicted_pileup = probabilities >= threshold

            pileup_efficiency = np.sum(
                predicted_pileup & true_pileup
            ) / n_pileup

            no_pileup_efficiency = np.sum(
                (~predicted_pileup) & true_no_pileup
            ) / n_no_pileup

            pileup_efficiencies.append(pileup_efficiency)
            no_pileup_efficiencies.append(no_pileup_efficiency)

        pileup_efficiencies = np.asarray(pileup_efficiencies)
        no_pileup_efficiencies = np.asarray(no_pileup_efficiencies)

        # Find maximum pileup efficiency satisfying the
        # minimum no-pileup efficiency requirement.
        valid = (
            no_pileup_efficiencies >= no_pileup_efficiency_threshold
        )

        if np.any(valid):

            valid_indices = np.where(valid)[0]

            best_index = valid_indices[
                np.argmax(pileup_efficiencies[valid])
            ]

            best_threshold = thresholds[best_index]
            best_pileup_efficiency = pileup_efficiencies[best_index]
            best_no_pileup_efficiency = no_pileup_efficiencies[best_index]

            print(
                f"Classification performance:\n"
                f"  Classification threshold: {best_threshold:.4f}\n"
                f"  Pileup efficiency:         {best_pileup_efficiency:.4f}\n"
                f"  No-pileup efficiency:      {best_no_pileup_efficiency:.4f}\n"
            )

        else:

            best_threshold = np.nan
            best_pileup_efficiency = np.nan
            best_no_pileup_efficiency = np.nan

            print(
                "No classification threshold satisfies the "
                f"no-pileup efficiency requirement of "
                f"{no_pileup_efficiency_threshold:.4f}"
            )

        # Plot efficiencies as a function of classification threshold.
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.scatter(
            thresholds,
            pileup_efficiencies,
            label="Pileup efficiency",
            s=200,
            color="firebrick"
        )

        ax.scatter(
            thresholds,
            no_pileup_efficiencies,
            label="No-pileup efficiency",
            s=200,
            color="royalblue"
        )

        ax.axhline(
            no_pileup_efficiency_threshold,
            color="black",
            linestyle="--",
            linewidth=5,
            label="Minimum no-pileup efficiency"
        )

        if np.isfinite(best_threshold):

            ax.scatter(
                best_threshold,
                best_pileup_efficiency,
                marker="*",
                s=500,
                color="black",
                zorder=5,
                label="Optimal operating point"
            )

            ax.axvline(
                best_threshold,
                color="black",
                linestyle=":",
                linewidth=2
            )

        ax.set_xlabel("Classification threshold")
        ax.set_ylabel("Efficiency")
        ax.set_title("Pileup classification performance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.outdir,
                f"classification_efficiency{self.endName}.png"
            ),
            dpi=150
        )

        plt.close(fig)

        return (
            best_pileup_efficiency,
            best_no_pileup_efficiency,
            best_threshold
        )


    def evaluate_pileup_decomposition(
        self,
        components,
        predicted_pileup,
        probabilities,
        threshold,
        ratio_bins=10,
        min_events_per_bin=1
    ):

        components = np.asarray(components, dtype=np.float32)
        predicted_pileup = np.asarray(predicted_pileup, dtype=bool)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        if components.ndim != 3:
            raise ValueError(
                f"Expected components with shape "
                f"(N, n_components, n_bins), got {components.shape}"
            )

        n_events, n_components, n_bins = components.shape

        if len(predicted_pileup) != n_events:
            raise ValueError("predicted_pileup length does not match components")

        if len(probabilities) != n_events:
            raise ValueError("probabilities length does not match components")

        # All events in this dataset are genuine pileup.
        pileup_efficiency = np.mean(predicted_pileup)

        print("\nPileup decomposition classification")
        print("-----------------------------------")
        print(f"Number of events:     {n_events}")
        print(f"Classification threshold: {threshold:.4f}")
        print(f"Pileup efficiency:    {100 * pileup_efficiency:.2f}%")

        component_amplitudes = np.max(components, axis=-1)

        component_positions = np.argmax(components, axis=-1)

        # Ignore zero-padded components.
        active_components = component_amplitudes > 0

        n_active = np.sum(active_components, axis=1)

        # We require at least two components for these metrics.
        valid_events = n_active >= 2

        min_separation = np.full(n_events, np.nan)
        max_amplitude_ratio = np.full(n_events, np.nan)

        for i in range(n_events):

            if not valid_events[i]:
                continue

            amplitudes = component_amplitudes[i, active_components[i]]
            positions = component_positions[i, active_components[i]]

            # Minimum separation between any two component maxima.
            sorted_positions = np.sort(positions)

            min_separation[i] = np.min(
                np.diff(sorted_positions)
            )

            # Largest component amplitude / smallest component amplitude.
            max_amplitude_ratio[i] = (
                np.max(amplitudes) / np.min(amplitudes)
            )

        print(f"Events with >= 2 components: {np.sum(valid_events)}")
        print(f"Events excluded:             {np.sum(~valid_events)}")

        # Only use events with valid component information.
        valid = (
            valid_events
            & np.isfinite(min_separation)
            & np.isfinite(max_amplitude_ratio)
        )

        def calculate_binned_efficiency(values, bin_edges):

            centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            efficiencies = np.full(len(centres), np.nan)
            uncertainties = np.full(len(centres), np.nan)
            counts = np.zeros(len(centres), dtype=int)

            for j in range(len(centres)):

                if j == len(centres) - 1:
                    in_bin = (
                        (values >= bin_edges[j])
                        & (values <= bin_edges[j + 1])
                    )
                else:
                    in_bin = (
                        (values >= bin_edges[j])
                        & (values < bin_edges[j + 1])
                    )

                counts[j] = np.sum(in_bin)

                if counts[j] >= min_events_per_bin:

                    efficiency = np.mean(
                        predicted_pileup[valid][in_bin]
                    )

                    efficiencies[j] = efficiency

                    # Binomial statistical uncertainty.
                    uncertainties[j] = np.sqrt(
                        efficiency * (1.0 - efficiency) / counts[j]
                    )

            return centres, efficiencies, uncertainties, counts

        distance_bin_edges = np.arange(
            0.5,
            n_bins + 0.5,
            1
        )

        (
            distance_centres,
            distance_efficiency,
            distance_uncertainty,
            distance_counts
        ) = calculate_binned_efficiency(
            min_separation[valid],
            distance_bin_edges
        )

        max_ratio = np.max(max_amplitude_ratio[valid])

        ratio_bin_edges = np.geomspace(
            1.0,
            max(1.01, max_ratio * 1.001),
            ratio_bins + 1
        )

        (
            ratio_centres,
            ratio_efficiency,
            ratio_uncertainty,
            ratio_counts
        ) = calculate_binned_efficiency(
            max_amplitude_ratio[valid],
            ratio_bin_edges
        )

        fig, ax = plt.subplots(figsize=(20, 20))

        ax.errorbar(
            distance_centres,
            distance_efficiency,
            yerr=distance_uncertainty,
            fmt="o",
            color="firebrick",
            capsize=16,
            capthick=3,
            linewidth=5,
            markersize=20,
        )

        ax.axhline(1, linestyle="--", linewidth=5, color="black")

        ax.set_xlabel("Minimum component separation [bins]")
        ax.set_ylabel("Pileup efficiency")
        ax.set_ylim(0.75, 1.05)
        ax.grid(alpha=0.3)

        ax.set_title(
            "Efficiency vs Component Separation"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.outdir,
                f"pileup_efficiency_vs_component_separation{self.endName}.png"
            ),
            dpi=150
        )

        plt.close(fig)

        fig, ax = plt.subplots(figsize=(20, 20))

        ax.errorbar(
            ratio_centres,
            ratio_efficiency,
            yerr=ratio_uncertainty,
            fmt="o",
            color="firebrick",
            capsize=16,
            capthick=3,
            linewidth=5,
            markersize=20,
        )

        ax.axhline(1, linestyle="--", linewidth=5, color="black")

        ax.set_xlabel("Maximum component amplitude ratio")
        ax.set_ylabel("Pileup efficiency")
        ax.set_title(
            f"Efficiency vs Amplitude Ratio"
        )
        ax.set_ylim(0.75, 1.05)
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.outdir,
                f"pileup_efficiency_vs_amplitude_ratio{self.endName}.png"
            ),
            dpi=150
        )

        plt.close(fig)

        return {
            "probabilities": probabilities,
            "predicted_pileup": predicted_pileup,
            "min_separation": min_separation,
            "max_amplitude_ratio": max_amplitude_ratio,
            "valid_events": valid,
            "pileup_efficiency": pileup_efficiency,
            "distance_centres": distance_centres,
            "distance_efficiency": distance_efficiency,
            "distance_uncertainty": distance_uncertainty,
            "distance_counts": distance_counts,
            "ratio_centres": ratio_centres,
            "ratio_efficiency": ratio_efficiency,
            "ratio_uncertainty": ratio_uncertainty,
            "ratio_counts": ratio_counts
        }



    # =========================================================
    # Plot decomposer fractional difference distributions
    # =========================================================

    def plot_parameter_performance(
        self,
        true_parameters,
        predicted_parameters,
        n_par_bins=10
    ):

        parameter_names = [
            "Amplitude",
            "Arrival time",
            "Rise time",
            "Fall time"
        ]

        parameter_units = [
            "Amplitude",
            "Bins",
            "Bins",
            "Bins"
        ]

        for iparam, name in enumerate(parameter_names):

            true_values = true_parameters[..., iparam].flatten()
            pred_values = predicted_parameters[..., iparam].flatten()

            valid = (
                np.isfinite(true_values)
                & np.isfinite(pred_values)
                & (np.abs(true_values) > 1e-8)
            )

            true_values = true_values[valid]
            pred_values = pred_values[valid]

            if len(true_values) == 0:
                print("No valid fits for", name)
                continue

            fractional_difference = (
                pred_values - true_values
            ) / true_values

            # Clip fractional differences to [-1, 1]
            fractional_difference = np.clip(
                fractional_difference,
                -1,
                1
            )

            # -------------------------------------------------
            # Fractional difference histogram
            # -------------------------------------------------

            fig, ax = plt.subplots(figsize=(20, 20))

            ax.hist(
                fractional_difference,
                bins=100,
                range=(-1, 1),
                histtype="step",
                linewidth=5,
                color="royalblue"
            )

            mean = np.mean(fractional_difference)
            std = np.std(fractional_difference)

            ax.axvline(
                mean,
                linestyle="--",
                linewidth=5,
                label=f"Mean = {mean:.3f}",
                color='black'
            )

            ax.set_xlabel(
                r"Fractional difference $(p_{\mathrm{pred}}-p_{\mathrm{true}})/p_{\mathrm{true}}$"
            )

            ax.set_ylabel("Entries")
            ax.set_title(name)
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    self.outdir,
                    f"fractional_difference_{name.lower().replace(' ', '_')}{self.endName}.png"
                ),
                dpi=150
            )

            plt.close(fig)

            # -------------------------------------------------
            # Mean and standard deviation versus true parameter
            # -------------------------------------------------

            # Use quantile bins to keep approximately equal
            # numbers of events in each bin.
            bin_edges = np.quantile(
                true_values,
                np.linspace(0, 1, n_par_bins + 1)
            )

            bin_edges = np.unique(bin_edges)

            bin_centres = []
            mean_values = []
            std_values = []

            for ibin in range(len(bin_edges) - 1):

                if ibin == len(bin_edges) - 2:

                    mask = (
                        (true_values >= bin_edges[ibin])
                        & (true_values <= bin_edges[ibin + 1])
                    )

                else:

                    mask = (
                        (true_values >= bin_edges[ibin])
                        & (true_values < bin_edges[ibin + 1])
                    )

                if np.sum(mask) < 2:
                    continue

                bin_centres.append(np.mean(true_values[mask]))

                mean_values.append(
                    np.mean(fractional_difference[mask])
                )

                std_values.append(
                    np.std(fractional_difference[mask])
                )

            bin_centres = np.asarray(bin_centres)
            mean_values = np.asarray(mean_values)
            std_values = np.asarray(std_values)

            fig, ax = plt.subplots(figsize=(20, 20))

            ax.errorbar(
                bin_centres,
                mean_values,
                yerr=std_values,
                fmt="o",
                capsize=16,
                capthick=3,
                linewidth=5,
                markersize=20,
                color="royalblue"
            )

            ax.axhline(0, linestyle="--", linewidth=5,color="black")

            ax.set_xlabel(
                f"True {name.lower()} ({parameter_units[iparam]})"
            )

            ax.set_ylabel("Mean fractional difference")
            ax.set_title(name)

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    self.outdir,
                    f"fractional_difference_mean_std_{name.lower().replace(' ', '_')}{self.endName}.png"
                ),
                dpi=150
            )

            plt.close(fig)

            print(
                f"{name:15s}: "
                f"mean = {mean:+.4f}, "
                f"std = {std:.4f}, "
                f"valid fits = {len(true_values)}"
            )

  

    def plot_component_similarity(
            self,
            true_components,
            predicted_components,
            similarity_scores=[0.95, 0.975, 0.98, 0.99, 0.999, 0.9999],
            n_par_bins=10
        ):

            # -------------------------------------------------
            # Calculate cosine similarity and nearest-neighbour
            # properties for each component
            # -------------------------------------------------

            similarities = []
            distance_similarities = []
            ratio_similarities = []

            nearest_distances = []
            amplitude_ratios = []

            similarity_event_indices = []
            similarity_component_indices = []

            n_samples, n_components, n_bins = true_components.shape

            for isample in range(n_samples):

                true_event = true_components[isample]
                pred_event = predicted_components[isample]

                # Identify non-empty, finite true components
                valid_components = (
                    np.all(np.isfinite(true_event), axis=1)
                    & (np.max(true_event, axis=1) > 1e-8)
                )

                component_indices = np.where(valid_components)[0]

                for icomp in component_indices:

                    true_comp = true_event[icomp]
                    pred_comp = pred_event[icomp]

                    if not np.all(np.isfinite(pred_comp)):
                        continue

                    true_norm = np.linalg.norm(true_comp)
                    pred_norm = np.linalg.norm(pred_comp)

                    if true_norm < 1e-8 or pred_norm < 1e-8:
                        continue

                    # Cosine similarity
                    similarity = np.dot(true_comp, pred_comp) / (
                        true_norm * pred_norm
                    )

                    similarity = np.clip(similarity, 0, 1.1)

                    similarities.append(similarity)
                    similarity_event_indices.append(isample)
                    similarity_component_indices.append(icomp)

                    # -------------------------------------------------
                    # Find closest neighbouring component
                    # -------------------------------------------------

                    other_indices = component_indices[
                        component_indices != icomp
                    ]

                    if len(other_indices) == 0:
                        continue

                    true_max_bin = np.argmax(true_comp)
                    true_max_amplitude = np.max(true_comp)

                    other_max_bins = np.argmax(
                        true_event[other_indices],
                        axis=1
                    )

                    other_max_amplitudes = np.max(
                        true_event[other_indices],
                        axis=1
                    )

                    distances = np.abs(
                        other_max_bins - true_max_bin
                    )

                    closest_index = np.argmin(distances)

                    nearest_distance = distances[closest_index]

                    closest_amplitude = other_max_amplitudes[
                        closest_index
                    ]

                    # Ratio of this component's peak amplitude
                    # to the closest component's peak amplitude
                    if closest_amplitude > 1e-8:

                        amplitude_ratio = (
                            true_max_amplitude / closest_amplitude
                        )

                        amplitude_ratios.append(amplitude_ratio)
                        ratio_similarities.append(similarity)

                        # Only include the distance when this component
                        # is smaller than its closest neighbour.
                        if amplitude_ratio < 0.75:

                            nearest_distances.append(nearest_distance)
                            distance_similarities.append(similarity)

            similarities = np.asarray(similarities)
            nearest_distances = np.asarray(nearest_distances)
            amplitude_ratios = np.asarray(amplitude_ratios)

            similarity_event_indices = np.asarray(
                similarity_event_indices
            )

            similarity_component_indices = np.asarray(
                similarity_component_indices
            )

            if len(similarities) == 0:
                print("No valid component comparisons")
                return

            # -------------------------------------------------
            # Histogram of cosine similarity
            # -------------------------------------------------

            fig, ax = plt.subplots(figsize=(20, 20))

            ax.hist(
                similarities,
                bins=100,
                range=(0.95, 1.01),
                histtype="step",
                linewidth=5,
                color="royalblue"
            )

            q16, median, q84 = np.percentile(
                similarities,
                [16, 50, 84]
            )

            ax.axvline(
                median,
                linestyle="--",
                linewidth=4,
                label=(
                    f"Median = {median:.4f}\n"
                    f"68% interval = [{q16:.4f}, {q84:.4f}]"
                ),
                color="black"
            )

            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Entries")
            ax.set_yscale("log")
            ax.set_title("Component similarity")
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    self.outdir,
                    f"component_similarity{self.endName}.png"
                ),
                dpi=150
            )

            plt.close(fig)

            # -------------------------------------------------
            # Helper function: median and quartiles versus a variable
            # -------------------------------------------------

            def plot_similarity_vs(
                x_values,
                y_values,
                xlabel,
                filename
            ):

                if len(x_values) == 0:
                    return

                bin_edges = np.quantile(
                    x_values,
                    np.linspace(0, 1, n_par_bins + 1)
                )

                bin_edges = np.unique(bin_edges)

                bin_centres = []
                median_values = []
                lower_errors = []
                upper_errors = []

                for ibin in range(len(bin_edges) - 1):

                    if ibin == len(bin_edges) - 2:

                        mask = (
                            (x_values >= bin_edges[ibin])
                            & (x_values <= bin_edges[ibin + 1])
                        )

                    else:

                        mask = (
                            (x_values >= bin_edges[ibin])
                            & (x_values < bin_edges[ibin + 1])
                        )

                    if np.sum(mask) < 2:
                        continue

                    values = y_values[mask]

                    q16, median, q84 = np.percentile(
                        values,
                        [16, 50, 84]
                    )

                    bin_centres.append(np.mean(x_values[mask]))
                    median_values.append(median)

                    lower_errors.append(median - q16)
                    upper_errors.append(q84 - median)

                bin_centres = np.asarray(bin_centres)
                median_values = np.asarray(median_values)

                asymmetric_errors = np.array([
                    lower_errors,
                    upper_errors
                ])

                fig, ax = plt.subplots(figsize=(20, 20))

                ax.errorbar(
                    bin_centres,
                    median_values,
                    yerr=asymmetric_errors,
                    fmt="o",
                    capsize=16,
                    capthick=3,
                    linewidth=5,
                    markersize=20,
                    color="royalblue"
                )

                ax.axhline(1, linestyle="--", linewidth=5, color="black")

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Median cosine similarity")
                ax.set_ylim(0.95, 1.01)
                ax.set_title("Component similarity")

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        self.outdir,
                        filename + self.endName + ".png"
                    ),
                    dpi=150
                )

                plt.close(fig)


            # -------------------------------------------------
            # Similarity versus distance to closest component
            # -------------------------------------------------

            plot_similarity_vs(
                np.asarray(nearest_distances),
                np.asarray(distance_similarities),
                "Distance to closest bigger component (bins)",
                "component_similarity_vs_distance"
            )

            # -------------------------------------------------
            # Similarity versus closest-component amplitude ratio
            # -------------------------------------------------

            plot_similarity_vs(
                np.asarray(amplitude_ratios),
                np.asarray(ratio_similarities),
                "Peak amplitude / closest other peak amplitude",
                "component_similarity_vs_amplitude_ratio"
            )

            # -------------------------------------------------
            # Plot true and predicted components at requested
            # similarity values
            # -------------------------------------------------

            n_plots = len(similarity_scores)

            if n_plots > 0:

                ncols = min(3, n_plots)
                nrows = int(np.ceil(n_plots / ncols))

                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(20 * ncols, 20 * nrows),
                    squeeze=False
                )

                for i, target_similarity in enumerate(similarity_scores):

                    ax = axes.flat[i]

                    # Find component with closest similarity
                    closest_idx = np.argmin(
                        np.abs(similarities - target_similarity)
                    )

                    actual_similarity = similarities[closest_idx]

                    event_idx = similarity_event_indices[closest_idx]
                    component_idx = similarity_component_indices[closest_idx]

                    true_comp = true_components[event_idx, component_idx]
                    pred_comp = predicted_components[event_idx, component_idx]

                    bins = np.arange(n_bins)

                    ax.hist(
                        self.edges[:-1],
                        bins=self.edges,
                        weights=true_comp,
                        histtype="step",
                        linestyle="-",
                        color="royalblue",
                        linewidth=5,
                        label="True component"
                    )

                    ax.hist(
                        self.edges[:-1],
                        bins=self.edges,
                        weights=pred_comp,
                        histtype="step",
                        linestyle="--",
                        color="firebrick",
                        linewidth=5,
                        label="Predicted component"
                    )

                    ax.set_title(
                        f"Target similarity = {target_similarity:.3f}\n"
                        f"Actual = {actual_similarity:.4f}, "
                        f"Event = {event_idx}, "
                        f"Component = {component_idx}"
                    )

                    ax.set_xlabel("Bin")
                    ax.set_ylabel("Amplitude")
                    ax.legend()

                    ax.grid(alpha=0.2)

                # Hide unused subplot axes
                for i in range(n_plots, nrows * ncols):
                    axes.flat[i].set_visible(False)

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        self.outdir,
                        f"component_similarity_examples{self.endName}.png"
                    ),
                    dpi=150
                )

                plt.close(fig)

            # -------------------------------------------------
            # Print summary
            # -------------------------------------------------

            max_idx = np.argmax(similarities)
            min_idx = np.argmin(similarities)

            print(
                f"Component similarity: "
                f"median = {median:.4f}, "
                f"68% interval = [{q16:.4f}, {q84:.4f}], "
                f"valid components = {len(similarities)}"
            )

            print(
                f"Highest similarity: event "
                f"{similarity_event_indices[max_idx]}, "
                f"similarity = {similarities[max_idx]:.4f}"
            )

            print(
                f"Lowest similarity: event "
                f"{similarity_event_indices[min_idx]}, "
                f"similarity = {similarities[min_idx]:.4f}"
            )

            return max_idx, min_idx


