`Contents <README.md>`__

Quick Start
===========

I just want to analyze data! Where should I put my script?
----------------------------------------------------------

If you are just hacking around, please put your code in ``scripts`` --
it's for any one-off analysis or snippet of code that isn't yet ready
for reuse by other people. We recommend using ``scripts/demo_script.py``
as a guide for the types of operations that you may find useful.

There is an `xforms <docs/xforms.md>`__ system for batching and saving
analyses in a way that you can reload later; this was what we used in
the ``scripts/fit_model.py`` command. We also have a good
tutorial/template at ``scripts/demo_script.py``. We recommend beginners
make a copy of it and edit it as needed. You may run it with:

::

    # Run the demo script
    python3 scripts/demo_script.py

The ``tests/`` directory is intended for unit tests that run
automatically with ``pytest``.

Where should I put my data? Where should I save my results?
-----------------------------------------------------------

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

    from nems.signal import Signal

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
