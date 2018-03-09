# Signals

`Signals` are the fundamental NEMS objects for storing timeseries data. The represent a value that changes in time, like the volume level of a speaker, the voltage across the membrane of a neuron, or the coordinates of a moving insect. 

`Signals` are objects designed to be useful for signal processing operations, slicing, averaging, jackknifing, truncating, splitting, saving/loading, dividing data into estimation and validation subsets, selecting parts of data where a condition is true, concatenating data sets together, and other common data wrangling tasks.

You can create `Signal` objects [from a file](#loading-signals-from-files), [from another signal](#creating-signals-from-other-signals), or [from a matrix](#creating-signals-from-numpy-arrays).

An important note for beginners: once created, a `Signal` object cannot be changed -- it is /immutable/, just like tuples in Python. This is intentional and enables specific optimizations while also preventing entire classes of errors. But don't worry: we have made it easy to create new `Signal` objects based on other ones.

## What is a signal?

Fundamentally, a `Signal` represents C channels being sampled for some period of time. Because NEMS is designed for signal processing, we usually assume that the sampling occurs T times at regular intervals, in which case the signal is "rasterized" at a particular sampling rate, and we represent the signal as an C x T array.

You may request this array using the `.as_continuous()` method of any signal.

A Signal can represent any type of data, such as subject pupil diameter, Local Field Potential (LFP), or a measurement of average neural spike rate during each discrete sampling period. 


## Loading Signals from files

The majority of the time, you will be loading a `Signal` stored on disk. At a minimum, you will need a CSV file (which holds the tabular data, like a 2D matrix) and a JSON file, which stores metadata and documents what kind of data is in the CSV file.

Optionally, you will also often have an "epochs" file, which helps tag interesting events or periods of time in the timeseries for later analysis, but we will defer [the detailed documentation of epochs](epochs.md) until later.

Loading a `Signal` from a file is trivial:
```
from nems.signal import Signal
sig = Signal.load('/path/to/my/signalfiles')
```

Creating your own CSV files is pretty straightforward, but you need to understand the format. Read on if that interests you, or [jump ahead if you would rather make Signal objects from a numpy array](#creating-signals-from-numpy-arrays).

### Example Signal CSV File

For this example, make a new directory in the `signals/` directory called `testrec` because we are pretending we made a test recording, and usually we group a collection of signals together into a data structure called a `Recording`.

Inside the `testrec` directory, make a file called `testrec_pupil.csv` and put the following CSV data inside it:

```
2.0, 2.1
2.5, 2.5
2.3, 2.3
2.4, 2.5
2.4, 2.3
2.3, 2.4
```

In the CSV file, each row represents an instant in time, and each column is a "channel" of information of this signal. Channels can be anything you want -- they are just there to help you group several dimensions together.

For this example, we'll pretend the first channel is the diameter of a test subject's left pupil and the second channel is their right eye. There are only two channels and six time samples here, but in many experiments you will have tens of channels and thousands or millions of time samples.

### Example Signal JSON File

Continuing our example, let's make a JSON file that describes the contents of the CSV file containing our pupil data.

In the `signals/testrec/` directory, make another file called `testrec_pupil.json` and fill it with:

```
{"recording": "testrec", "name": "pupil", "chans": ["left_eye", "right_eye"], "fs": 0.1, "meta": {"Subject": "Don Quixote", "Age": 36}}
```

Here,

- `recording` is the name of the recording. We group collections of signals into "recordings", which is just a name to help us group simulatneously recorded signals. 
- `name` is the name by which you want to refer to this signal. Generally it should match your file name so as not to be confusing.
- `fs` is the sampling rate in Hz. Generally it will be 10, 50, or even 44,200Hz, but for our test example, we assume that a measurement of the pupil diameter was only taken every 10 seconds, so `fs=0.1`.
- `chans` is the name of each channel (i.e. column in the CSV file), from left to right. 
- `meta` is extra information about the recording, such as the time of day it was taken, the experimenter, the subject, their age, or other relevant information. You may place anything you want here as long as it is a valid JSON data structure.

### Loading Example CSV + JSON

Assuming that your signal directory looks like this:

```
├── signals
│   └── testrec
│       ├── testrec_pupil.csv
│       └── testrec_pupil.json
```

You should now be able to load the pupil signal by creating a file at `scripts/pupil_analysis.py` with the contents:

```
from nems.signal import Signal

# Note that we don't append the suffix .json or .csv
# because we are loading two files simultaneously
sig = Signal.load('signals/testrec/testrec_pupil')
```

And launch it from your terminal with:

```
cd /path/to/nems
python3 scripts/pupil_analysis.py
```

That's it! You can start using your `Signal` now. Read on to find a short guide to interesting operations that you can do with a Signal. 

## Creating Signals from Other Signals

It's really common to make one signal from another signal. At the moment, we have a variety of methods that are rather in development flux, but the ones that produce new signals include:

```
    def normalized_by_mean(self):
    def normalized_by_bounds(self):
    def split_at_time(self, fraction):
    def jackknifed_by_epochs(self, epoch_name, nsplits, split_idx, invert=False):
    def jackknifed_by_time(self, nsplits, split_idx, invert=False):
    def concatenate_time(cls, signals):
    def concatenate_channels(cls, signals):
```

TODO: Link to python-generated documentation here. 

## Creating Signals from Numpy Arrays

This technique for creating signals is most common when importing or loading data from a custom format. In general, we encourage you to avoid saving your data in custom formats so that data files are more easily shared, but if you have special needs, then writing your own custom signal loader or subclass of `Signal` is completely acceptable.

```
from nems.signal import Signal

numpy_array = load_my_custom_data_format(...)

# Not shown here, but we suggest using optional arguments "epochs" and "meta"
# as well as recording, name, matrix, and fs.
sig = Signal(recording='my_recording_name',
             name='my_signal_name',
             matrix=numpy_array,
             fs=200)

# Optional: save it as a signal for next time or for easy sharing
sig.save('../signals/my-new-signal')
```

## Signal Subclasses

We will now discuss two subclasses of signals that can be useful to reduce data storage on disk, but are otherwise functionally identical. 

### Subclass: EventSignals

Now, the signal processing view of a `Signal` is "external" view that we actually use during signal processing. However, as the sampling rate gets faster and faster, the C x T representation of a Signal becomes more and more wasteful. For events that occur only occasionally, we can save space if we store a list of discrete event times, rather than having a matrix of mostly zeros with only a few ones. 

In this case, we use a subclass of the `Signal` object called an `EventsSignal`, which may be rastered into time bins at any sampling frequency desired, and then used as a normal Signal from there on.


### Subclass: RepeatedSignal

A second special case occurs, for example, when we have stimuli that are repeated tens or hundreds of times. While such a stimulus can certainly be represented with a C x T array, it is again a wasteful representation. 

In such cases, the `RepeatedSignal` subclass of the `Signal` object is useful. Rather than store a large raster, it stores a single copy of each unique event and rasterizes it only as requested. 

For example, say we have a P-channel spectrogram and several different stimuli of different lengths S_1, S_2, etc. The `RepetitiveSignal` internally stores a `{name1: [C x S_1], name2: [C x S_2], ...}` dictionary, in which the keys are the names/labels of the stimuli and the [C x S_*] arrays are what to insert. 

The `RepetitiveSignal` object thus rasterizes signals on demand by using a signal`s `.epochs` datastructure and the `.replace_epochs()` function to produce a C x T matrix only when needed.


## Closing Thoughts on Signals

If you want to have a model that uses the data from 100 neurons, you can either have a single 100-channel Signal, or 100 one-channel signals. It's up to you.

Future work tickets:

- TODO: Subclassed Signal that rasterizes from an spike time list
- TODO: Prototype how signals can implement the numpy interface (See next section)

### Signals Implement the Numpy Interface

Signals implement the Numpy universal function interface. This means that you can perform a variety of array operations on Signals:

    # Add a DC offset of 5 to the signal
    offset_signal = signal + 5

    # Matrix multiplication
    weighted_channels = weights @ signal

    # Multi-signal operations (stim and pupil are signals)
    pred = stim * pupil + stim * pupil**2 + stim * pupil**3

    # Apply a linear filter to the signal. A new signal is created as fir
    fir = lfilter(b, a, stim)

    # Now, average across the filtered channels.
    fir_mean = fir.mean(axis=0)

When performing an operation on a signal, a new signal object is returned. The signal will be identical to the original object, albeit with appropriately-transformed data (e.g., sampling rate and epochs will be copied over). 

If you attempt to perform an operation (e.g., adding two signals) that do not match in some attribute (e.g., number of samples, sampling rate, etc.) you'll get an error.
