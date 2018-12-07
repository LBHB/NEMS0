# Installing NEMS #

## Generic Python Install

Download NEMS:
```
git clone https://github.com/lbhb/NEMS
```
Add the NEMS library via pip (where "NEMS" is the installation directory):
```
pip install -e NEMS
```
NEMS libraries should now be loadable. See next section for how to try it out!

## Conda installation

Conda is simple and popular platform for Python package management. NEMS is does not currently have a 
conda package, but you can use conda to manage python and your other packages.
To install NEMS in a conda environment, first
download and install Conda from here:
[https://www.anaconda.com/download/](https://www.anaconda.com/download/).

Create a NEMS environment:
```
conda create -n NEMS
conda activate NEMS # or in Windows: activate NEMS
```
Install required packages and some useful utilities:
```
conda install ipython pip jupyter numpy scipy matplotlib pandas requests h5py sqlalchemy
```
Download NEMS:
```
git clone https://github.com/lbhb/NEMS
```
Then add the NEMS library via pip (where "NEMS" is the installation directory):
```
pip install -e NEMS
```

## Other installation options

We have found that Python 3 distributions compiled with the Intel MKL libraries are about twice as fast 
as the default Python implementations that come installed on many linux machines. 
Please see our (slightly outdated) [conda installation instructions](docs/conda.md) if you 
would like to set up a python environment like the way that we do. NOTE: NEMS is designed to use Python 3. Backwards compatibility with Python 2 is untested and unsupported.

Alternatively (not recommended), you may install all the dependencies on your own, e.g.,
```
pip install requests numpy scipy matplotlib pandas sqlalchemy h5py
```

You may want to add NEMS to your python path. Eg, in Linux:
```
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/NEMS\"" >> ~/.bashrc
source ~/.bashrc
```

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
