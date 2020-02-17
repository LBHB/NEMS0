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

**NWB**: Support for reading datasets stored in Neurodata Without Borders (NWB) format. Pip install the allensdk packages as part of NEMS. You can run this either before or after the
generic install described above:

::

    pip install -e ./NEMS/[nwb]

**Tensorflow**: If you're using conda to manage packages, it's best to install Tensorflow via conda:

::
    # without GPU (run on CPU):
    conda install tensorflow
    # or with GPU support:
    conda install tensorflow-gpu

If you're using pip, install Tensorflow support with pip:

::

    pip install -e ./NEMS/[tensorflow]

**Sphinx** (if you want to be able to edit and regenerate documentation):

::

    pip install -e ./NEMS/[docs]

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
