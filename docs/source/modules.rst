Modules
=======

Module perform a single transformation in the
:doc:`modelspec <modelspecs>` functional cascade. They should contain
only string and numerical values (not functions themselves) so that they
can be converted to JSON format for saving.

Each module is described by a dictionary with several key/value pairs.
Some are required:

**Required**

- ``fn``: pathspec to function that performs the transformation (e.g., ``nems.modules.fir.basic``)
  This function should accept a :doc:`Recording object <recording>` and return a dictionary of
  signals, typically only signals that were modified or created by ``fn``. These signals will be merged
  into the recording and passed to the next module in the :doc:`modelspec <modelspecs>` cascade.

- ``fn_kwargs``: dictionary of fixed args passed to fn. Can be empty, but typically contains:

  - ``i``: name of input signal
  - ``o``: name of output signal
  - ``s`` name of state signal

- ``phi``: dictionary of free parameters, concatenated with ``fn_kwargs`` and passed to ``fn`` for
  evaluation. These parameters are updated during fitting.

**Optional**

- ``plot_fns``
- ``plot_fn_idx``
- ``priors``

The only *mandatory* field in each module dict is ``fn``. All other fields
are optional. Presently, we have reserved the following fields for
specific uses:

+-------------+--------+--------------------------------------------------------+
| Field       | Type   | Description                                            |
+=============+========+========================================================+
| id          | str    | Name of the module keyword that created this dict.     |
+-------------+--------+--------------------------------------------------------+
| fn          | str    | The pure transformation function.                      |
+-------------+--------+--------------------------------------------------------+
| fn_kwargs   | dict   | Non-fittable arguments that are passed to fn.          |
+-------------+--------+--------------------------------------------------------+
| meta        | dict   | A dictionary of metadata, descriptions, benchmarks,    |
|             |        | etc                                                    |
+-------------+--------+--------------------------------------------------------+
| phi         | dict   | Like fn, but fittable. The "parameters" of the module. |
+-------------+--------+--------------------------------------------------------+
| prior       | dict   | Defines a prior distribution from which bounds or      |
+-------------+--------+--------------------------------------------------------+
|             |        | samples may be drawn by the fitter during              |
|             |        | optimization.                                          |
+-------------+--------+--------------------------------------------------------+

You may place whatever information at all you like in ``meta`` without
consulting anyone -- "description", "date", "notes", or information on
fitters and data files are all good types of things to put in the
``meta`` dict. However, in the interests of forcing helpful design
discussions, if you wish to establish a convention for another top-level
field for modules, please bring it up in the `NEMS Gitter
channel <https://gitter.im/lbhb/nems>`__


Making your own module
----------------------
TODO
