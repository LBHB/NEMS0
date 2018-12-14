# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical 
model to time series data, plotting the model predictions, and comparing the predictive 
accuracy of multiple models. We use it to develop and test [computational models of how
sound is encoded in the brains of behaving mammals](https://hearingbrain.org), but it will 
probably work with your timeseries data as well.

[Main Help](docs/README.md)

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

