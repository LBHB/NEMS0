# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to study [computational models of how auditory stimuli are neurally encoded in mammalian brains](https://hearingbrain.org), but it will probably work with your timeseries data as well.


## Installation

```
git clone https://github.com/lbhb/nems

# Please make sure that the nems repo is in your path. This works for linux:
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/nems\"" >> ~/.bashrc
source ~/.bashrc
```

We have found that Python 3 distributions compiled with the Intel MKL libraries are about twice as the default Python implementations that come installed on many linux machines. Please see our basic [conda installation instructions](conda.md) if you would like to set up a python environment like the way that we do.

Finally, if for some reason they are not installed already, you will also need to use pip to install several libraries that NEMS uses:

```
pip install requests numpy scipy matplotlib pandas
```

## Your First Model Fit

You may test if everything is working by telling NEMS to download some sample auditory stimulus-response data, use a simple linear-nonlinear model (which should taking about 2 minutes to fit), and then save the results locally:

```
cd nems/scripts
INPUT_URI=https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/TAR010c-18-1.tar.gz
MODELKEYWORDS="wc18x1_lvl1_fir15x1_dexp1"
DESTINATION=~/nems_results/
python fit_model.py $INPUT_URI $MODELKEYWORDS $DESTINATION
```

Now you should go in `~/nems_results` and find the subdirectory where the plot PNG files, modelspecs, and other files were saved. In our laboratory, we have a [NEMS_DB](http://github.com/lbhb/nems_db) server that we save our fit models to, so the destination is an URL like `http://ourserver.edu/results`. If you will be saving, searching, and sharing lots of model fits with other people in your lab, you may want to consider installing NEMS_DB as well.


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
5. Detailed Guides
   - [Architectural Diagram](docs/architecture.svg)
   - Creating your own modules
   - Comparing your models with others
   - Sharing modules, models, and data with others
6. Contributing to NEMS
   - [How To Contribute](docs/contributing.md)
   - [Design Discussion Archive](docs/discussions.md)
   - [Development History](docs/history.md)
