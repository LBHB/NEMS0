Overview
========

Input Data
~~~~~~~~~~

If your data is in `Recording <recordings.md>`__ format, put it in the
``recordings/`` directory. If it is not yet in recording form, you may
want to make a conversion script in ``scripts/`` that is able convert
your data from your custom format into `Signals <signals.md>`__ and
`Recordings <recordings.md>`__.

Below is some pseudocode for a script that converts your data from a
custom format and then save it as a Signal to make it easier for other
people to use.

::

    from nems0.signal import Signal

    numpy_array = load_my_custom_data_format(...)
    sig = Signal(matrix=numpy_array, 
                 name='mysignal',
                 recording='some-string', 
                 fs=200  # Hz
                 )
    sig.save('../recordings/my-new-recording/my-new-signal')

Output Results
~~~~~~~~~~~~~~

Once you have fit a model to a recording, you can save the resulting
files either locally in your ``nems/results`` directory, or to a remote
`database <database.md>`__ server.

Load data
~~~~~~~~~

Models are fit using datasets packaged as
`recordings <recordings.md>`__, which are a collection of
`signals <signals.md>`__, typically at least a stimulus and response.

1. Get some demo data with ``nems.recording.get_demo_data``

2. Load recording from file saved in NEMS native format.
   ``nems.recording.load_recording``

3. Generate a recording on the fly, e.g., with
   ``nems.recording.load_recording_from_arrays``

Preprocess the data
~~~~~~~~~~~~~~~~~~~

1. Mask out "bad" epochs

2. Generate state signals from epochs

3. Transform the stimulus(?)

Define a modelspec
~~~~~~~~~~~~~~~~~~

The `modelspec <modelspecs.md>`__ defines a sequence of transformations.
Each transformation is called a `module <modules.md>`__, and applies a
transformation to a signal that models some stage of neural processing.
Modules include linear reweighting, FIR filter, static nonlinearity,
etc. To define the modelspec:

1. Use keywords

   -  ``modelspec=nems.initializers.from_keywords(model_keyword_string)``

   -  default keywords in ``nems.plugins.default_keywords``

   -  see `keywords.md <>`__

2. Assemble the modelspec as a list of modules

Perform the model fit
~~~~~~~~~~~~~~~~~~~~~

1. Load parameters from a simpler, previously fit model

2. Define jack-knife subsets for n-fold cross-validation

3. "Pre-fit" a subset of the model parameters (``nems.initializers``)

4. Fit a state-independent model

5. Fit a fully state dependent model

Evaluate the model
~~~~~~~~~~~~~~~~~~

Plot the results
~~~~~~~~~~~~~~~~

Save results
~~~~~~~~~~~~

Reload for analysis later
~~~~~~~~~~~~~~~~~~~~~~~~~
