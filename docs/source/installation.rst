Installing NEMS
===============

Generic Python Install
----------------------

If you already have Python and Git, download NEMS:

::

   git clone https://github.com/lbhb/NEMS

Add the NEMS library via pip (where ``./NEMS`` is the installation directory and ``-e`` means editable mode,
which is not necessary, but is useful if you end up customizing your model fits):

::

   pip install -e ./NEMS

NEMS libraries should now be loadable!

Conda Installation
------------------

Conda is simple and popular platform for Python package management. NEMS
does not currently have a conda package, but you can use conda to
manage Python and your other packages. To install NEMS in a conda
environment, first download and install Conda from `here <https://www.anaconda.com/download/>`__. Git should
be installed with Conda.

Download NEMS:

::

    git clone https://github.com/lbhb/NEMS

Create a NEMS environment, then activate it and install NEMS:

::

    conda env create -n NEMS -f ./NEMS/environment.yml
    conda activate NEMS
    pip install -e ./NEMS


Extra dependencies
------------------

NWB support. After installing nems, install the allensdk packages:

::

    cd NEMS
    pip install -e .[nwb]

Tensorflow, after installing NEMS, install Tensorflow
::

    cd NEMS
    pip install -e .[tensorflow]

(Or, even better, install Tensorflow via conda before installing NEMS.)

Notes
-----

We have found that Python 3 distributions compiled with the Intel MKL
libraries are about twice as fast as the default Python implementations
that come installed on many linux machines.

NEMS is designed to use Python 3. Backwards compatibility with Python 2
is untested and unsupported.

You may want to add NEMS to your python path. Eg, in Linux:

::

    echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`/NEMS\"" >> ~/.bashrc
    source ~/.bashrc
