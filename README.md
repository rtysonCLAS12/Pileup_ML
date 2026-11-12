# Pileup_ML

Code to train ML models to identify drift chamber pileup events and predict individual waveforms contributing to pileup events.

## Requirements

Install required libraries using venvs and pip in a new environment called newenvname (you should choose a better name):

      python3 -m venv /path/to/env/location/newenvname
      source /path/to/env/location/newenvname/bin/activate.csh
      pip install torch 
      pip install matplotlib
      pip install scipy
      pip install scikit-learn
      pip install lightning
      pip list

Remember to always activate your environment before running code with source /path/to/env/location/newenvname/bin/activate.csh .

## Cheat Sheet

```
python3 saveAndPlot.py            #produce toy datasets
python3 trainClassifier.py        #train pileup vs no-pileup classifier
python3 testClassifier            #test classifier performance
python3 trainDecomposer.py        #train model to predict individual waveforms
python3 testDecomposer.py         #test decomposer performance
```

## Toy Pileup Generator

The `ToyPileUpGenerator` class generates synthetic drift-chamber waveforms for training and evaluating machine-learning models for pileup classification and waveform decomposition. Individual detector pulses are modelled using a normalised difference of exponentials, with randomly sampled amplitudes, arrival times, rise times and fall times. Multiple pulses are summed to produce pileup events, with an additional event-level baseline offset and Gaussian electronic noise.

### Waveform Parameters

The waveform parameters are independently sampled from uniform distributions within configurable ranges. The default values are:

| Parameter | Default range | Description |
|---|---|---|
| `amplitude_range` | `(0.5, 5.0)` | Pulse peak amplitude |
| `arrival_time_range` | `(-2.0, 18.0)` | Pulse arrival time (time bins) |
| `rise_time_range` | `(0.8, 2.5)` | Pulse rise-time constant |
| `fall_time_range` | `(3.0, 9.0)` | Pulse decay-time constant |
| `noise_std` | `0.03` | Gaussian electronic noise standard deviation |
| `baseline_std` | `0.02` | Event-level baseline offset standard deviation |

The waveform sampling and pileup multiplicity can also be configured through `n_bins` (default: 30) and `max_n_waveforms` (default: 2). The fall-time constant is constrained to be at least 0.5 time units greater than the rise-time constant.

### Configuration

All parameters can be configured when initialising the generator:

```python
generator = ToyPileUpGenerator(
    n_bins=30,
    max_n_waveforms=3,
    noise_std=0.03,
    baseline_std=0.02,
    amplitude_range=(0.5, 5.0),
    rise_time_range=(0.8, 2.5),
    fall_time_range=(3.0, 9.0),
    arrival_time_range=(-2.0, 18.0),
    seed=42
)
```

### Dataset Generation

Two dataset generation methods are available:

- **`generate_classification_dataset()`**: Generates events containing a randomly selected number of pulses (1 to `max_n_waveforms`), labelled as either single-pulse (`0`) or pileup (`1`).
- **`generate_decomposition_dataset()`**: Generates events containing a fixed number of pulses, with the noisy summed waveform as input and the individual underlying waveforms as targets.

Both methods automatically split the generated data into training and testing datasets and save them as compressed `.npz` files. The random seed can be specified to ensure reproducibility.

The saveAndPlot.py script is set-up to generate 1M events for each type of dataset with appropriate initial parameters, save the datasets to appropriate directories in the current working directory, and plot a few examples of the generated dataset.


## Classifier Training

The classifier model is contained in PileUpClassifier. This is a simple 1D CNN binary classifier to predict if an event contains or not pileup. The trainClassifier.py will run the model training on the dataset generated in the previous section. The training script is set-up such that the training will stop if the validation binary cross-entropy loss plateaus. The model with the best validation loss is saved, and the script plots the training and validation loss as a function of epoch.

The testClassifier.py script applies the classifier to the testing dataset. The script plots the output of the classifier, and the pileup and no-pileup efficiency as a function of the threshold on the output of the classifier. The decomposition dataset containing only pileup events is loaded in and the classifier is applied to this dataset. The pileup efficiency is then plotted as a function of the distance between two closest waveform peaks, and the ratio of the amplitude of the peak. 

In general, the model is found to perform better as the separation between waveform increases, and for waveforms of similar relative amplitude.

## Decomposition Training

The decomposer model is contained in ComponentDecomposer. This is a simple 1D CNN that predicts the individual waveforms contributing to a pileup event. The trainDecomposer.py will run the model training on the dataset generated by the toy generator. The training script is set-up such that the training will stop if the validation loss plateaus. The model with the best validation loss is saved, and the script plots the training and validation loss as a function of epoch. Note that the loss used for training is the MSE loss between predicted and true component. The sum of the predicted and true components are also compared and the MSE between these along with the sum of both MSEs are recorded and plotted as a function of epoch. 

The testDecomposer.py script applies the decomposer to the testing dataset. The script plots examples of the predicted individual waveforms, compared to the true waveforms. The function used to generate the waveforms are fitted to the true and generated waveforms. The fractional difference between the predicted and true waveform fit parameters is plotted as a histogram, and the mean and standard deviation of the fractional difference is plotted as a function of the fit parameters. The cosine similarity between true and predicted waveforms is obtained and plotted as a histogram. The median and 68% interval is plotted as a function of the distance between two closest waveform peaks, and the ratio of the amplitude of the peak. 

In general, the model is found to perform better as the separation between waveform increases, and for waveforms of similar relative amplitude.

