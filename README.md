# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to study [computational models of how auditory stimuli are neurally encoded in mammalian brains](https://hearingbrain.org), but it will probably work with your timeseries data as well.

## Installation

```
git clone https://github.com/lbhb/nems

# Please make sure that the nems repo is in your path. This works for linux:
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/nems\"" >> ~/.bashrc
source ~/.bashrc
```

We have a good tutorial/template at `scripts/demo_script.py`. We recommend beginners make a copy of it and edit it as needed. You may run it with:

```
# Run the demo script
python3 scripts/demo_script.py
```

## Table of Contents ## 

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
