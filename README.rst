.. image:: https://travis-ci.com/LBHB/NEMS.svg?branch=master
    :target: https://travis-ci.com/LBHB/NEMS
    :alt: Documentation Status

.. image:: https://readthedocs.org/projects/nems/badge/
    :target: https://nems.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/LBHB/NEMS
    :target: https://github.com/LBHB/NEMS/blob/master/LICENSE.txt
    :alt: GitHub license

NEMS
====

NEMS is the Neural Encoding Model System. It is helpful for fitting a
mathematical model to time series data, plotting the model predictions,
and comparing the predictive accuracy of multiple models. We use it to
develop and test `computational models of how sound is encoded in the
brains of behaving mammals <https://hearingbrain.org>`__, but it will
work with many different types of timeseries data.

Docs
----

Full documentation can be found `here <https://nems.readthedocs.io>`__.

We also have a `Gitter chat <https://gitter.im/lbhb/nems>`__ where you can get help from other users.

Installation
------------

If you don’t already have Python and Git installed, see the `installation
instructions <https://nems.readthedocs.io/en/latest/installation.html>`__ for recommended procedures.

Quick Install
~~~~~~~~~~~~~

If you already have Python and Git, download NEMS:

::

   git clone https://github.com/lbhb/NEMS

Add the NEMS library via pip (where ``./NEMS`` is the installation directory and ``-e`` means editable mode,
which is not necessary, but is useful if you end up customizing your model fits):

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

   cd NEMS/scripts
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

Click on ``demo_xforms.pynb`` and give it a whirl!
