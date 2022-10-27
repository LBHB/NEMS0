.. image:: https://travis-ci.com/LBHB/NEMS.svg?branch=master
    :target: https://travis-ci.com/LBHB/NEMS
    :alt: Documentation Status

.. image:: https://readthedocs.org/projects/nems/badge/
    :target: https://nems.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/LBHB/NEMS
    :target: https://github.com/LBHB/NEMS/blob/master/LICENSE.txt
    :alt: GitHub license

NOTICE!!!!!
===========


This repository is being phased out and replaced by a "lite" and much more user-friendly version of NEMS at https://github.com/LBHB/NEMS. If you need to maintain functionality of the orginal nems tools, you should install the nems_db libary, found here: <https://github.com/LBHB/nems_db/tree/nems0>


**THIS REPOSITORY IS NO LONGER BEING MAINTAINED**

Please don't use this repository unless you know what you're doing
==================================================================


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

Direct link to the some `demos <https://nems.readthedocs.io/en/latest/demos/demos.html>`__. We also have
a `Gitter chat <https://gitter.im/lbhb/nems>`__ where you can get help from other users.

Installation
------------

If you don’t already have Python and Git installed, see the `installation
instructions <https://nems.readthedocs.io/en/latest/installation.html>`__ for recommended procedures.

Quick Install
~~~~~~~~~~~~~

If you already have Python and Git, download NEMS:

::

   git clone https://github.com/lbhb/NEMS

Add the NEMS library via pip (where ``./NEMS`` is the installation directory and ``-e`` means editable mode):

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

Click on ``demo_xforms.ipynb`` and give it a whirl!
