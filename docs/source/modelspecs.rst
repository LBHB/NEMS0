Modelspecs and Keywords
=======================

What's a ModelSpec?
-------------------

A modelspec is a data structure that unambiguously defines a sequence of
transformations in a NEMS model. It is a simple, minimalist format that
is easily :doc:`saved and loaded to disk <saving>`.

A modelspec is essentially an ordered list of dicts. Each dict describes
a single :doc:`module <modules>` that performs an input-output
transformation performed by a pure function (``fn``). An example of a
simple modelspec before fitting is:

::

    [{"id": "wc18x1",
      "fn": "nems.modules.weight_channels.weight_channels",
      "fn_kwargs": {"input": "stim-spectrogram",
                   "output": "pred"},
      "prior": [TODO]},
      {"id": "fir10x1",
      "fn": "nems.modules.fir.fir_filter",
      "fn_kwargs": {"input": "pred",
                   "output": "pred"},
      "prior": [TODO]},
     {"id": "dexp1"
      "fn": "nems.modules.nonlinearity.double_exponential",
      "fn_kwargs": {"input": "pred",
                    "output": "pred"},
      "prior": [TODO]}]

After fitting, that same modelspec might look like this:

::

    [{"id": "wc18x1",
      "fn": "nems.modules.weight_channels.weight_channels",
      "fn_kwargs": {"input": "stim-wav",
                   "output": "pred"},
      "phi": {"coefficients": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]},
      "prior": [TODO],
      "meta": {"initializers": ["prefitwc3"],
               "fit_start": "2018-02-02T20:33Z",
               "fit_end": "2018-02-02T29:14Z",
               "datasets": ["por39c-39", "gus038a-a2"],
               "fitters_used": ["iterfit39", "adagrad2"]}},
      {"id": "fir10x1",
      "fn": "nems.modules.fir.fir_filter",
      "fn_kwargs": {"input": "pred",
                   "output": "pred"},
      "phi": {"coefficients": [[0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]]},
      "prior": [TODO]},
     {"id": "dexp1"
      "fn": "nems.modules.nonlinearity.double_exponential",
      "fn_kwargs": {"input": "pred",
                    "output": "pred"},
      "phi": {"amplitude": 1.0,
              "shift": 1.0,
              "base": 1.0,
              "kappa": 1.0},
      "prior": [TODO]}]

More on the :doc:`modules <modules>` that comprise modelspecs here.

What's an Initializer? What are module keywords?
------------------------------------------------

Often, we will be reusing the same module dicts over and over again,
with the only thing that changes being the parameter values ``phi``
which resulted from fitting the model to some specific data. Rather than
write out the modelspec from scratch every single time, we can use an
``initializer`` function to create a modelspec for us.

If you know the shape of the data coming in, you may directly create a
modelspec using short abbreviations called "keywords". For example, we
could have created the above modelspec using:

::

    my_modelspec = initializers.from_keywords('wc18x1_fir10x1_dexp1')

The ``from_keywords`` initializer accepts an underscore separated string
to make a list of keywords, and then each keyword is replaced with a
dict that was looked up in ``nems.keywords.defaults``. (You may use your
own dictionary if you wish however, using the ``registry={...}``
argument.)

This is a very simple initializer to be sure, but it has several
advantages:

1) Every time we see ``fir10x1``, we know that it maps to a modelspec
   with a FIR filter that has a known shape -- in this case, 10 time
   bins by 1 channel. This makes studying parameters simpler than if we
   used ``fir10`` and did not know the number of input channels. In such
   a case, to find the matrices that we want, we would need to look at
   all of the ``fir10`` objects, determine the shape of the coefficients
   matrix, and then discard those that don't match what we wanted.
   Conversely, we immediately know that ``fir10x1`` is not the same as
   ``fir10x2`` because they have different numbers of channels and the
   keywords are not identical.

2) If ``fir10x1`` is `saved in the modelspec
   filename <#how-do-you-save-or-load-a-modelspec>`__, you can easily
   find all filenames containing this keyword, and easily extract/merge
   their contents to determine the distribution of post-fit parameter
   values.

3) Sometimes the same function is used in multiple ways, and module
   keywords can help provide metadata about the way in which it was
   intended to be used. For example, we may use two FIR filters in a
   single model, one of which uses the "active" part of behavioral data
   and the other which makes predictions on the "passive" behavioral
   data. By using two keywords, ``fir10x1active`` and
   ``fir10x1passive``, it is much easier for us to determine the
   function of each of the filter at a later time, without complicated
   inspection of the modelspec.

More Complex Initializers
-------------------------

What if you know the overally structure of the model we want to fit, but
not the shape of the data coming in, and you want to adjust the model's
keywords to the data?

Most of the time, you can just adapt the first keyword. For example,
"wc14x1" might become "wc18x1", meaning the that input data is expected
to be a 14-channel spectrogram or an 18-channel spectrogram.

For more unusual initializations, you may need to write your own
initializer function. This function can then study the data's shape or
values, it may accept arguments you need to define the "rough shape" of
the model, and finally generates a modelspec.

It is perfectly acceptable (and recommended!) for one initializer to
call another initializer. For example, in this case, we might call
``nems.initializers.from_keywords()`` after looking at the incoming
data's dimensionality and then deciding what keywords to use. In another
case, we might look at the behavioral data and decide if we needed to
use keywords corresponding to "active/passive" conditions or
"reference/probe' conditions. Initializers may be specific to certain
experimental types, for example.

*Recommendation*: Please try to preserve the 1-to-1 mappings created by
the module keywords shorthands. One way to do this is to make the your
custom initializer also use the defaults keywords registry. This
preserves our ability to search quickly through modelspecs to find ones
containing keyword ids or parameters of interest, while also having the
convenience of quickly generating models of a certain type.

Who decides what the keywords mean?
-----------------------------------

The default keyword registry is defined in ``nems/keywords.py``. It is
the place for "stable" keywords that are unlikely to be changed further.

During development, we recommend making your own personal keywords
registry, and combining it with the defaults registry when creating
modelspecs. Later, once your keywords are more stable, they may be
migrated into the default registry.

For example,

.. code-block:: python

    import defaults import nems.keywords.api

    my_registry = {'mork1': {'fn': 'nems.modules.mork.spork',
                             'api': 'weight_channels',
                             'fn_kwargs': {},
                             'prior': [],
                             'phi': {}},

    merged_registry = defaults.append(my_registry)

    my_modelspec = initializers.from_keywords('wc18x1_mork1_dexp1',
        registry=merged_registry)

How do you save or load a modelspec?
------------------------------------

``nems/modelspec.py`` contains useful functions for loading and saving
modelspecs in files. The four functions of interest are:

.. code-block:: python

    save_modelspec()   # Saves a single modelspec to a single file
    save_modelspecs()  # Saves a list of related modelspecs to multiple files
    load_modelspec()   # Load a single modelspec from a single file.
    load_modelspecs()  # Loads multiple (related) modelspecs from multiple files

These simple functions are mostly to encourage uniform pattern for model
and file names. You may override the default file name if desired, but
for compatability, the NEMS defaults for a model are generated using:

::

    1. The keyword string that define the modelspec
    4. The fitter used to find the parameters
    2. The shorthand name of Recording object used to fit model parameters
    3. The date and time, in ISO8601 format (Suggestion: 2018-02-02T19:02Z)

Allowed But Not Always Recommended: Prefitters
----------------------------------------------

Because initializers are just functions, there is no limit to the
operations you may perform when generating a modelspec. If necessary,
you might write a ``nems.initializers.fir_prefit`` initializing function
for this purpose to loosely prefit the filter parameters (or even
priors) to your data set. Because the initializer need not be saved with
the modelspec, it need not be run again, and so loading the model at a
later time will not have any increased performance penalties.

However, as an alternative to "baking in" this computation implicitly
into a single initializer or keyword, we would instead recommend putting
effort towards creating iterative or multi-stage fitting algorithms that
work for all or most models. Fitters are more easily shared between a
wide variety of models than initializers that usually are connected with
specific keywords.

Future Work: Preprocessors in the Model
---------------------------------------

Our current strategy for performing preprocessing will be to use
parameter-free modules and then to cache the results using memoization
of those modules (probably via joblib).

TODO.

Under Debate: What additions to the modelspec have not yet been decided?
------------------------------------------------------------------------

Items in discussion:

0. What should the keyword convention be? Last number is # of channels?
   Should there be any?
1. Should ``keywords`` be called "nicknames" instead? Or does nobody
   care?
2. Should keywords be generated from many small individual files so that
   we can track changes in git? Or is this 'defaults' and 'private'
   dictionary approach sufficient for now?
3. What should the "default" filename for models be?
4. Where should the "fitter" metadata be appended? Are the metedata
   properties of a modelspec the superset of all of the modules?

Priors
------

*Priors*: I pushed support for initializing phi from priors to the dev
branch today. There are three functions that return modified (copies) of
modelspecs with the phi initialized from the priors.

.. code-block:: python


    new_modelspec = nems.priors.set_mean_phi(modelspec)
    # or
    new_modelspec = nems.priors.set_random_phi(modelspec)
    # or
    new_modelspec = nems.priors.set_percentile_phi(modelspec, 0.1)

A value of phi initialized using the idea of specific and general
preferences:

1. Prefer a phi parameter already set in the module;

2. Otherwise, generate any uninitialized phi parameters from the
   ``prior`` of that module, if one exists;

3. Otherwise, fall back on priors defined the ``default_priors`` data
   structure to make any remaining uninitialized phi parameters.

You may mix and match. If you look at ``keywords.py`` below, you can see
that I manually set the initial value of 'amplitude', manually define a
prior for ``base``, and let the the 'shift' and 'kappa' values be set by
default.



.. code-block:: python

    defaults = {
        'wc40x1': {
            'fn': 'nems.modules.weight_channels.weight_channels',
            'fn_kwargs': {
                'i': 'stim',
                'o': 'pred'
            },
            'phi': {
                'coefficients': [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0]
                ]
            }
        },
        'fir10x1': {
            'fn': 'nems.modules.fir.fir_filter',
            'fn_kwargs': {
                'i': 'pred',
                'o': 'pred'
            },
            'phi': {
                'coefficients': [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            }
        },
        'dexp1': {
            'fn': 'nems.modules.nonlinearity.double_exponential',
            'fn_kwargs': {
                'i': 'pred',
                'o': 'pred'
            },
            'phi': {'amplitude': 2.0},
            'prior': {'base': ('Normal', [0, 10])}
        }
    }

If not specified in the modelspec, these priors will be used
============================================================



.. code-block:: python

    default_priors = {
        'nems.modules.fir.fir_filter': {
            'coefficients': ('Normal', [
                [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            ])
        },
        'nems.modules.nonlinearity.double_exponential': {
            'base': ('Normal', [0, 1]),
            'amplitude': ('HalfNormal', [0.5, 0.5]),
            'shift': ('Normal', [0, 1]),
            'kappa': ('HalfNormal', [0.5, 0.5])
        }
    }


Note that in general, the size of the priors determine the size of
``phi``. The exception to this is ``default_priors`` which should always
be 1D so that people can use those very vague values as starting places
for custom initializations with initializers.
