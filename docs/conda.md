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

I (Ivar) had bad luck using conda to install packages in the environment before activating the environment. This should have worked but did not:

```
conda install numpy scipy pandas matplotlib 
```

This is probably due to my inexperience with conda. The workaround is simply to use `pip` to install once I was in the conda environment.