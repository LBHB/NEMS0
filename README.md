# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to study [computational models of how auditory stimuli are neurally encoded in mammalian brains](https://hearingbrain.org), but it will probably work with your timeseries data as well.


## Installation

```
git clone https://github.com/lbhb/NEMS

```

Then add the NEMS library via pip (where "NEMS" is the installation directory):
```
pip install -e NEMS
```
We have found that Python 3 distributions compiled with the Intel MKL libraries are about twice as fast as the default Python implementations that come installed on many linux machines. Please see our basic [conda installation instructions](docs/conda.md) if you would like to set up a python environment like the way that we do.

NOTE: Regardless of which Python distribution you choose to use, NEMS is designed to use Python 3. Backwards compatibility with Python 2 is untested and unsupported.

Alternatively (not recommended), you may install all the depdencies on your own, e.g.,

```
pip install requests numpy scipy matplotlib pandas
```
Then add NEMS to your python path. Eg, in Linux:
```
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/NEMS\"" >> ~/.bashrc
source ~/.bashrc
```

## Your First Model Fit

You may test if everything is working by telling NEMS to download some sample auditory stimulus-response data, use a simple linear-nonlinear model (which should taking about 2 minutes to fit), and then save the results locally:

```
cd NEMS/scripts
ipython

[0]: run demo_script.py
```

Open `demo_script.py` in an editor to work through each step of the fit.


## Table of Contents ##

This documentation is a work in progress as of March 9, 2018.

1. [Quick Start](docs/quickstart.md)
2. Organizing your Data
   - [Signals](docs/signals.md)
   - [Recordings](docs/recordings.md)
   - [Epochs](docs/epochs.md)
3. Organizing your Models
   - [Modules](docs/modules.md)
   - [Modelspecs](docs/modelspecs.md)
   - [Distributions](docs/distributions.ipynb)
4. Fitting your Models
   - [Priors](docs/priors.md) -- TODO
   - [Fitters](docs/fitters.md) -- TODO
   - [Xforms](docs/xforms.md)
5. Detailed Guides
   - [Architectural Diagram](docs/architecture.svg)
   - Creating your own modules
   - Comparing your models with others
   - Sharing modules, models, and data with others
6. Contributing to NEMS
   - [How To Contribute](docs/contributing.md)
   - [Design Discussion Archive](docs/discussions.md)
   - [Development History](docs/history.md)
7. Other
   - [NGINX Caching](docs/nginx.md)
