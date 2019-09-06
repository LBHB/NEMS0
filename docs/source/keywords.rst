KEYWORDS
========

What are they used for?
-----------------------

Keywords are used in conjunction with NEMS' xforms sytem to briefly and
uniquely define a full model - including functions for loading,
preprocessing, and fitting - with a single string. A complete set of
keywords, such as ``ozgf.fs100.ch18_wc.15x2-fir.2x15-dexp.2_basic.cd``,
is often referred to as a modelname. The LBHB team takes advantage of
this to be able to quickly indicate a full model to fit from a terminal,
as well as to faciliate querying results by modelname in our database.

How are they interpreted?
-------------------------

There are several layers within a modelname designated by special
characters.

Each modelname is separated into three groups, separated by underscores:
loaders, modules, and fitters. All modelnames will contain this
separation, but the separations described below may or may not be
present.

Within each group, there may be an arbitrary number of keywords
separated by hyphens.

Within each keyword, there are an arbitrary number of options separated
by periods.

Within each option, there may be commas to separate arguments where
appropriate (between indices, for example).

Within an option that can contain floating point parameters, a 'd' will
be interpreted as a decimal. For example, the tolerance set T3,5d5,7
would be translated as [3.0, 5.5, 7.0].

An example modelname containing all of these characters might be:

::

    ld.n_wc.18x1.g-fir.1x18-dexp.1_init.t5d5.t5-iter.cd.ti50.fi20.T3,5,7.S0,1

Which would be translated as:

-  Load a recording (uri specified elsewhere) and apply normalization.
-  Apply gaussian channel weighting.
-  Apply a basic FIR filter.
-  Apply double exponential output nonlinearity.
-  Use fit\_basic\_init to initialize phi for the model, with a
   tolerance level of 10^-5.5.
-  Use fit\_iteratively with coordinate descent to fit the model. Use
   tolerance levels 10^-3, then 10^-5, then 10^-7 with 50 iterations per
   tolerance level and 20 iterations per fit loop. Only apply fitting to
   modules 0 and 1.

For a full description of how individual keywords are parsed, refer to
their definition within the modules in nems.plugins.
