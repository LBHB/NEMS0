NOTICE!!
========

NEMS0 is an archived version of the Neural Encoding Model System (NEMS). This repository is being phased out and replaced by a "lite" and more user-friendly version of NEMS at https://github.com/LBHB/NEMS.

However, we are maintaining this repository to support analysis of some published data.


NEMS
====

NEMS is the Neural Encoding Model System. It is helpful for fitting a
mathematical model to time series data, plotting the model predictions,
and comparing the predictive accuracy of multiple models. We use it to
develop and test `computational models of how sound is encoded in the
brains of behaving mammals <https://hearingbrain.org>`__, but it will
work with many different types of timeseries data.


Installation
------------

Requirements
~~~~~~~~~~~~

Installing NEMS requires python (tested with version 3.7) and git. We recommend using conda to create an environment
specifically for NEMS. More recent versions of python are likely to work, but the Quick Install below may not work
out of the box, as you may need to make sure that the various dependencies have compatible versions.

::

    conda create -n nems python=3.7


Quick Install
~~~~~~~~~~~~~

Once you have python and git installed, download NEMS:

::

   git clone https://github.com/lbhb/NEMS0

If using conda, make sure you have activated your NEMS environment. Then add the NEMS0 library via pip (where ``./NEMS`` is the installation directory and ``-e`` means editable mode):

::

   pip install -e ./NEMS

NEMS libraries should now be loadable. See next section for how to try it out!

Your First Model Fit
--------------------

Via Python Console
~~~~~~~~~~~~~~~~~~

You may test if everything is working by telling NEMS to download some
sample auditory stimulus-response data, use a simple linear-nonlinear
model (which should taking about 2 minutes to fit), and then save the
results locally:

::

   cd NEMS0/scripts
   ipython

   In [1]: run demo_script.py

Or open ``demo_script.py`` in an editor to work through each step of
the fit.

Via Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~

If you have Jupyter installed:

::

   cd NEMS/notebooks
   jupter notebook

Click on ``demo_xforms.ipynb`` and give it a whirl!
