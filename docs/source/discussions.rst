Design Discussions
==================

The following are a list of design discussion points that were brought
up in our Slack conversations and may help inform why the design ended
up the way that they did. It is probably not useful to non-developers.

DB ACCESS
---------

Direct mysql DB access from NEMS is forbidden; instead, assume you have
the files on disk (they can be fetched via HTTP, jerbs, DAT, or
whatever)

MODELSPECS
----------

The more I think about it, the more I wonder if our real goal should
just be to exorcise all state from a module, turning modules into "pure
functions" (a.k.a. functions without side effects). Not that we
shouldn't use classes, but that we should keep focused on stateless
functions when possible.

Managing modules can be complicated precisely because they contain
mutable state. Given that state is usually easier when it is all in once
place, maybe packing the entire model into a single data structure isn't
such a crazy idea.

The following shows a little demo of how that might look in general, and
for three cases that are not supported by the current version of NEMS:
1. "Per-module fitters", when each module uses a different sub-fitter
and all modules are iterated through. 2. "Parameter fitting dropout",
when some parameters are randomly NOT considered for optimization during
the fitting process 3. "Module dropout", when the transformation of
random modules are temporarily omitted during the fitting process

The latter two techniques have been used extensively in deep learning
and allegedly make the fitting process more robust. And per-module
fitters could be a cleaner way of putting all the fitting code in a
single place rather than spreading it out and putting a little in each
module, which then requires that other fitters remove manually whenever
they do not want to do an "initial-fit".

MODELSPECS AND PHI
------------------

Q: Why is Phi part of the modelspec? I have to clone the modelspec each
time I want to run a different fit on the model!

A: I'm conceptually thinking about modelspecs as being something that
defines the entire input-output "black box" model; yes, the parameters
(phi) are a special case in many cases, but they still fall within the
black box and can't be logically separated from it without having to lug
around the knowledge that this phi goes with that black box, and this
other phi goes with that other black box. I'm willing to pay the very
slight extra memory use because I think we can optimize it away in other
ways.

KEYWORDS
--------

Our previous implementations of "keywords" were functions that could do
anything. Now they are discouraged, because they were doing too many
things at once: A) Appending of a module to a model; B) Initializing
module parameters, which is better done with a JSON that describes prior
beliefs about the values; C) Fitting module parameters. Our new strategy
should be to use keywords as abbreviations that help create a modelspec.

MODULES MAY NOT INTROSPECT THE STACK
------------------------------------

Modules are now forbidden from knowing anything about the rest of the
stack. If you can think of a good reason why they absolutely need to see
the rest of the stack, bring it up ASAP; otherwise we are going to plan
to disallow this. Yes, this implies that STRF plots (which need multiple
models) should be done by a function that takes in a model as an
argument, not by a module method.

SIGNAL DIMENSIONS
-----------------

I'd like to propose a simplification. In my opinion, the simplest and
most general case for all data is for Signal objects to always be 2D:
channel x time\_sample, like a CSV file with each row being a moment in
time. Trial-by-trial groupings shouldn't matter for the evaluation of a
module (I hope?), and not supporting 4D arrays gets rid of nasty
problems that occur at the boundaries of a convolved signal when it is
folded by trial. Basically, then, if you want to have a model that uses
the data from 100 neurons, you can either have a 100 channel Signal, or
100 one-channel signals. It's up to you.

SPLITTING DATA
--------------

I don't think data splitting should be part of the Fitter class -- data
selection is a judgement call by the programmer before the fitting
begins. You may want to use 3 data files as estimation data, and then
use one for validation data. Or, you may want to use 60% of a single
file as estimation data, and then 40% as validation. It really varies,
and depends on the analysis. Also, some fitters accidentally have used
the entire data set, not just the estimation data set; we should try to
avoid this class of problems by keeping splitting out of the fitter.

FITTERS
-------

This is the hardest part in my experience and needs the most thought.
Just to muddy the waters a bit, here are some things that came up in the
past: - People inadvertently cheating by using the whole data set for
fitting, instead of just the estimation data set. - Fitting roughly at
first with one algorithm, then fitting with another to get a final fit
(it's stupid, but it works better) - Iterating over all modules, fitting
only parameters from one module at a time (it's stupid, but it works
often) - Fitting subsets of the parameters (to avoid n^2 performance
penalties with some fitters) - Plugging in different cost functions for
the same fitter (L1 vs L2 vs Log Likelihood) - Using one cost function
for fitting, but multiple cost functions (metrics) for evaluating the
final performance - Trying different termination conditions. Usually a
predicate function that returns true when you should stop fitting.
Reasons to stop fitting include a certain number of model evaluations,
gradient step size, average change in error, too many NaN predictions,
or elapsed time. - A helpful performance optimization in NARF was to
avoid recomputing the entire stack; only recompute modules whose
parameters were changed or had previous modules with changed parameters.

PRIORS AS SETTING BOUNDS
------------------------

Every Module implements a get\_priors method that takes one argument
(the data being fit). The module may use this data to help come up with
reasonable information about the fit bounds. For each parameter, a
distribution is returned. This distribution defines the min/max and
range of possible values. - So, for a value that can take on any
positive values, you'd use a Gamma distribution. The mean of the gamma
distribution (E[x] = alpha/beta) will be set to what you think is a
reasonable value for that parameter. The fitter can then choose to set
the initial value for the parameter to E[x] or draw a random sample from
the distribution. - For a value that can take on any value, you'd use a
Normal centered at what a reasonable expected starting point for the
value is. - For a value that must fall between 0 and 1 you can choose
either a Uniform or Beta distribution. - For parameters that are
multi-dimensional (e.g. FIR coefficients and mu in weight channels), the
Priors can be multidimensional as well. So, for weight channels you can
specify that the first channel is a Beta distribution such that the
channel most likely falls at the lower end of the frequency axis and the
second channel at the upper end of the frequency axis.

SUM\_CHANNELS
-------------

Should this be renamed "sum\_channels.py"? We might have a
'sum\_signals.py" module at some point. Also: should this summing
implementation be put in the "signals" object, which we then call from
this file, in order that we don't accidentally have two
similar-but-not-identical versions of the same code? (I guess the answer
to this depends on whether signals are passed between modules or not, as
the same problem comes up with a "normalization" module and the
Signal.normalize() methods)

Fitter Input Argument Specs
---------------------------

I think I may be arguing with my past self here, but I am wondering if
we can remove the need to pass the "model" object to our fitting
algorithms? I would ideally just prefer to have fitters accept a cost
function, instead of having any knowledge about the model structure. I
feel like any optimizations (evaluating part of the stack, per-module
fitters) could still be accomplished with carefully structured
functional composition.

Inter-module Data Exchange Format
---------------------------------

Now that we have Signal objects, have we decided the data type once and
for all? Numpy arrays? Or Signal/Recording objects? The former is
probably more efficient, the latter is (debatably) more convenient for
interoperability. Since the signal object was not available before, I
can see that Brad assumed numpy arrays would be exchanged -- is that
necessary for Theano to work?

Lazy Devil's Advocate
---------------------

Q: To rethink a design decision, is it really worth wrapping all of the
scipy.stats distributions with nems.distributions.\* instead of instead
of using them directly? What specific advantages do we get from this?

A: It's easier for us to control the behavior if we wrap the
distributions. For example, look at
nems.distributions.distributions:Distribution.sample. It's not just a
simple mapping to the underlying scipy.stats distribution.

SCIPY
-----

I have functional versions of the modules, fitters and model portions of
the system right now. To see how we can implement it using a bayes
approach vs scipy, compare nems.fitters.scipy and nems.fitters.pymc3.
The bayes approach is a very abstract system and requires quite a bit of
knowledge re how PyMC3 (the bayes fitting package) works, so I haven't
documented it in depth. Basically PyMC3 uses symbolic computation to
build a symbolic model, then evaluates it once it's built.

ITERATIVE FITS
--------------

Stephen's very concerned about "mini-fits", so the iterative\_fit
function in the nems.fitters.scipy should hopefully alleviate his
concerns.

FUNCTIONAL FITTERS
------------------

I've made the fitting routines functions (i.e., functional approach)
rather than objects. It just seems to make more sense for these basic
fits. There's no reason why some fitters can't be objects (e.g., if we
are building a complex fitter with sub-fitters for each module and we
need a central object to track the state).

ON THE NAMES OF FUNCTIONS
-------------------------

To help with clarity, we will define the following words mathematically:

::

     |-----------+----------------------------------------------------------|
     | Name      | Function Signature and Description                       |
     |-----------+----------------------------------------------------------|
     | EVALUATOR | f(mspec, data) -> pred                                   |
     |           | Makes a prediction based on the model and data.          |
     |-----------+----------------------------------------------------------|
     | METRIC    | f(pred) -> error                                         |
     |           | Evaluates the accuracy of the prediction.                |
     |-----------+----------------------------------------------------------|
     | FITTER    | f(mspec, cost_fn) -> mspec                               |
     |           | Tests various points and finds a better modelspec.       |
     |-----------+----------------------------------------------------------|
     | COST_FN   | f(mspec) -> error                                        |
     |           | A function that gives the cost (error) of each mspec.    |
     |           | Often uses curried EST dataset, METRIC() and EVALUATOR() |
     |-----------+----------------------------------------------------------|

    where:
       data       is a dict of signals, like {'stim': ..., 'resp': ..., ...}
       pred       is a dict of signals, just like 'data' but containing 'pred'
       mspec      is the modelspec data type, as was defined above
       error      is a (scalar) measurement of the error between pred and resp

WHERE SHOULD THE DATASPEC BE RECORDED?
--------------------------------------

TODO: Open question: even though it is only a few lines, how and where
should this information be recorded? The data set that a model was
trained on is relevant information that should be serialized and
recorded somewhere.

::

     save_to_disk('/some/path/model1_dataspec.json',
                  json.dumps({'URL': URL, 'est_val_split_frac': 0.8}))

TODO: This annotation should be done automatically when split\_at\_time
is called?

Splitting, Jackknifing, and Epochs
----------------------------------

@jacob In reply to your excellent question about what we should do for
jackknifed\_by\_epochs and splitting based on epochs, and what data
formats those should return, I think I made a mistake in asking for
regex matching as part of the core functionality, and I'd like to walk
that back a bit.

On the dev branch, I basically just removed the "regex" matching from
split\_at\_epoch things, and things just worked fine. I didn't fix
jackknife\_by\_epochs yet, and I'm not entirely sure what the right way
to do that is, and I'm open to ideas. My current hunch is to make it
more like jackknife\_by\_time, and I'm guessing that rounding to the
nearest occurence of by\_epochs would be the way to do it (and warn if
the rounding is off results in partitions that, say, differ more than
some critical amount). But I'm open to ideas.

Now, I still think regex functionality is cool, but after talking with
SVD, I'm thinking we should do that in a single function, like
``signal.match_epochs('regex')`` which will give us a list of all
matching epochnames that we can then iterate through.

Something like:

::

    TORCs = signal.match_epochs('^TORC.*')

    for torc in TORCs:
        my3dmatrix = signal.fold_by(torc)
        mean_for_this_torc = numpy.mean(my3dmatrix, axis=0)
        plot(mean_for_this_torc)

Mostly, I just wanted to avoid 4D arrays since they make my head hurt
when they become ragged or partially NaN'd in strange ways.

--------------

I know we have gone over some of these points before, but I wanted to
write down some of the things Stephen, Charlie, and I just discussed so
that Jake and Brad have a chance to give their input as well.

We focused on Charlie's analysis, which largely rests on analyzing the
data and slicing it in unusual ways. It's a good test case for the
Signal/Recording stuff we have been working on. The data is >2000
seconds long, so sampled at 100Hz, the data matrix has more than 200,000
time samples.

*START\_TIME VS START\_INDEX*. We really need to get signals and epochs
using absolute time and not bin indexes!

*DATA EXCISION*. One of the analyses is to find the average response to
each stimulus. Some stimuli only occur 3 times, and are only 5.5 seconds
long (550 samples). Right now, ``fold_by('birdhonk.wav')`` leaves you
with a matrix that is 3 x C x T, where T is very large. We really need
an argument to fold\_by() that makes it excise data, so that we can make
the output matrix be 3 x C x 550.

*MULTI\_FOLD\_BY*. I'm not sure what to call this, so please suggest a
better name. ``fold_by()`` returns a 3D matrix, but what we need in
several cases is to produce a dictionary in which the keys are the names
of stimuli and the values are 3D matrices produced by ``fold_by()`` with
excision. Something as simple as:

``def multi_fold_by(signal, list_of_epoch_names):     d = {ep : signal.fold_by(epoch_name) for epin list_of_epoch_names}     return d``

*INVERSE MULTI\_FOLD\_BY OPERATION*. Another operation that we need is
the inverse of ``multi_fold_by()``. That is, a way of building up a
Signal object from a dict of 2D matrices (C x T) and an epochs data
structure. This has two applications: 1) Creating a rasterized stimuli
from a ``stim_dict (test_parmread.py: Line 46)`` and some epochs for
when to start the stimuli 2) After using ``mega_fold_by()`` and
averaging away the first dimension, in order to create a signal that is
the same size as the original response, but has the 'average' response
to every signal of a particular kind.

*SUBCLASS OF SIGNAL*. One thing that would also clearly be useful is a
subclass of Signals that internally represents the data as unrasterized
spike times, to save space, and then rasterizes it on demand to produce
the matrix you want. This also gives us a 'canonical form' of our data,
since you can produce many rasters from a single spike-time list. We
agreed that it's simplest just to raster everything for now, because our
modules work on rastered data, and then at some point in the future we
subclass ``Signal`` and store data internally in a different way.

*PREPROCESSING*. We have also had discussions about "preprocessing" vs
models. One crazy idea is to use a model with zero fittable parameters
to do preprocessing, so that you can preprocess signals in an
unambigious way. Then you feed those preprocessed signals into another
model and do your fitting on that second model like normal.

IMMUTABILITY OF EPOCHS. Right now, epochs are mutable because they are
panda dataframes, but the rest of the Signal is immutable. In the
future, if we want to test the equality of two signals, this is easiest
if they are completely immutable because we can just test the references
instead of testing every substructure of the data.

OCCURRENCES vs REPETITIONS. A thought as we standardize our terminology
and home in on best practice for signals and epochs. I suggest we use
the word "occurrences" rather than "repetitions" when the number of
times an epoch appears in a signal. To me, "repetitions" implies that
each one is repeated/identical. This is fine for stimuli, but not true
for responses. On the other hand, "occurrences" is not specific as to
whether the occurrences are identical or not. Does that sound good?

Re: a comment in demo\_script.py
``# TODO: temporary hack to avoid errors resulting from epochs not being defined. #for signal in rec.signals.values(): #    signal.epochs = signal.trial_epochs_from_reps(nreps=10) # If there isn't a 'pred' signal yet, copy over 'stim' as the starting point. # TODO: still getting a key error for 'pred' in fit_basic when #       calling lambda on metric. Not sure why, since it's explicitly added.``
I'm going to remove these; the former doesn't appear to be causing
errors anymore, and the latter I think should be handled with explicit
keywords. (see modelspecs.md, I just wrote it today) (edited)

--------------

TODO: @Ivar -- per architecture.svg looked like this was going to be
====================================================================

handled inside an analysis by a segmentor? Designed fit\_basic with
===================================================================

that in mind, so maybe this doesn't go here anymore, or I may have
==================================================================

had the wrong interpretation. --jacob
=====================================

TODO: @Ivar -- Raised question in fit\_basic of whether fitter should be
========================================================================

exposed as argument to the analysis. Looks like that may have been
==================================================================

your original intention here? But I think if the fitter is exposed,
===================================================================

then the FitSpaceMapper also needs to be exposed since the type of
==================================================================

mapping needed may change depending on which fitter is use.
===========================================================

These are both great questions that I am only just now getting to. I
think yes, we handle the segmentation inside the analysis, and that as
drawn in architecture.svg, we just have "data" and "modelspec" as the
only two /required/ arguments to an analysis. However, it also totally
makes sense to have /optional/ arguments for the segmentor, mapper, cost
function, and anything else we come up with.

--------------

TODO: @Ivar -- per architecture.svg looked like this was going to be
====================================================================

handled inside an analysis by a segmentor? Designed fit\_basic with
===================================================================

that in mind, so maybe this doesn't go here anymore, or I may have
==================================================================

had the wrong interpretation. --jacob
=====================================

Yes, we will probably make two analyses at some point:

1. The outer analysis, which segments the data into a est and val
   dataset
2. The inner analysis, which may or may not not segment the est dataset
   during the fitting process.

But for the moment, we'll leave the outer loop in demo\_script.py.
------------------------------------------------------------------

Ideas on initializers:

Initializers are like: f(data, incomplete\_modelspec) ->
modelspec\_with\_priors. Or maybe f(data, parameters) ->
modelspec\_with\_priors, where the parameters could be either 'vague'
keywords or whatever needed?

The goal is that after initialization, when fitting is ready to start,
we have a modelspec containing priors and keywords that help us find
this model later.

There will be many kinds of initializers: if you have a particularly
weird model, you may want to prefit it in some weird way. If you want to
start from another model, you might use the "start near this existing
fit model" initializer. I leave it up to everybody to make their own
initializers for specifically hard problems. Otherwise keywords may
"just work" for simpler things.
