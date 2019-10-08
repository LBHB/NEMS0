Installing NEMS
===============

Generic Python Install
----------------------

Download NEMS, where /my/code/path/ is where you keep your code:

::

    git clone -b dev https://github.com/lbhb/NEMS /my/code/path/

Add the NEMS library via pip (where "NEMS" is the installation
directory):

::

    pip install -e /my/code/path/NEMS

NEMS libraries should now be loadable. See next section for how to try
it out!

Conda installation
------------------

Conda is simple and popular platform for Python package management. NEMS
is does not currently have a conda package, but you can use conda to
manage python and your other packages. To install NEMS in a conda
environment, first download and install Conda from here:
https://www.anaconda.com/download/.

Create a NEMS environment:

::

    conda create -n NEMS -f ./nems/environment.yml

Then activate that environment and install NEMS, as above:

::

    conda activate NEMS # or in Windows: activate NEMS
    git clone -b dev https://github.com/lbhb/NEMS /my/code/path/
    pip install -e /my/code/path/NEMS


Other installation options
--------------------------

We have found that Python 3 distributions compiled with the Intel MKL
libraries are about twice as fast as the default Python implementations
that come installed on many linux machines. Please see our (slightly
outdated) `conda installation instructions <docs/conda.md>`__ if you
would like to set up a python environment like the way that we do. NOTE:
NEMS is designed to use Python 3. Backwards compatibility with Python 2
is untested and unsupported.

Alternatively (not recommended), you may install all the dependencies on
your own, e.g.,

::

    pip install requests numpy scipy matplotlib pandas sqlalchemy h5py

You may want to add NEMS to your python path. Eg, in Linux:

::

    echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/NEMS\"" >> ~/.bashrc
    source ~/.bashrc
