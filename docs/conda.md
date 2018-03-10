# Conda and NEMS

Although NEMS will work with most python3 implementations, we recommend using the conda package manager to create an environment compiled against Intel MKL library and the most recent version of numpy. In our performance tests, we have found that use of these libraries can make NEMS run nearly *twice* as fast as the python that comes installed on most linux distributions. 

```
# Get and install conda from:
https://conda.io/miniconda.html

# Set up an environment
conda create -n nemsenv python=3 -c intel

# Get inside the environment
source activate nemsenv

# Install any needed packages inside that
pip install -e nems

```
