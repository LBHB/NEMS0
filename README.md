# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical 
model to time series data, plotting the model predictions, and comparing the predictive 
accuracy of multiple models. We use it to develop and test [computational models of how
sound is encoded in the brains of behaving mammals](https://hearingbrain.org), but it will 
probably work with your timeseries data as well.

## Installation

If you don't already have Python installed, see the 
[installation instructions](docs/installation.md)
 for recommended procedures.  

### Quick Generic Python Install

If you already have Python, download NEMS:
```
git clone https://github.com/lbhb/NEMS
```
Add the NEMS library via pip (where "NEMS" is the installation directory):
```
pip install -e NEMS
```
NEMS libraries should now be loadable. See next section for how to try it out!

## Your First Model Fit

### Via python console

You may test if everything is working by telling NEMS to download some sample auditory stimulus-response data, use a simple linear-nonlinear model (which should taking about 2 minutes to fit), and then save the results locally:

```
cd NEMS/scripts
ipython

In [1]: run demo_script.py
```
And/or open `demo_script.py` in an editor to work through each step of the fit.

### Via jupyter notebook

If you have jupyter notebook installed:
```
cd NEMS/notebooks
jupter notebook
``` 
Click on `demo_xforms.pynb` and give it a whirl!

## Table of Contents ##

This documentation is a work in progress as of Dec 1, 2018.

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
