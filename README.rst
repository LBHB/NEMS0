NEMS
====

Full documentation here:
  .. image:: https://readthedocs.org/projects/nems/badge/?version=dev
    :target: http://nems.readthedocs.io/en/dev/?badget=latest
    :alt: Documentation Status

NEMS is the Neural Encoding Model System. It is helpful for fitting a
mathematical model to time series data, plotting the model predictions,
and comparing the predictive accuracy of multiple models. We use it to
develop and test `computational models of how sound is encoded in the
brains of behaving mammals <https://hearingbrain.org>`__, but it will
work with many different types of timeseries data.

We also have a Gitter chat where you can get help from other users: https://gitter.im/lbhb/nems

Installation
------------

If you don’t already have Python installed, see the `installation
instructions <https://nems.readthedocs.io/en/dev/installation.html>`__ for recommended procedures.

Quick Generic Python Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have Python, download NEMS. Currently we recommend using
the ``dev`` branch, which contains a number of useful features:

::

   git clone -b dev https://github.com/lbhb/NEMS

Add the NEMS library via pip (where “NEMS” is the installation
directory):

::

   pip install -e NEMS

NEMS libraries should now be loadable. See next section for how to try
it out!

Your First Model Fit
--------------------

Via python console
~~~~~~~~~~~~~~~~~~

You may test if everything is working by telling NEMS to download some
sample auditory stimulus-response data, use a simple linear-nonlinear
model (which should taking about 2 minutes to fit), and then save the
results locally:

::

   cd NEMS/scripts
   ipython

   In [1]: run demo_script.py

And/or open ``demo_script.py`` in an editor to work through each step of
the fit.

Via jupyter notebook
~~~~~~~~~~~~~~~~~~~~

If you have jupyter notebook installed:

::

   cd NEMS/notebooks
   jupter notebook

Click on ``demo_xforms.pynb`` and give it a whirl!
