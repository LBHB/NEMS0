# SVD Priorities for getting NEMS working by early March 2018 #

Updating Mar 4, 2018

Primary goal is ot replicate the functionatility of NEMS first version. 
Easiest thing is to list a few use cases in a bit more detail.

Update Mar 8, 2018

xforms is the answer??
1. xforms.save_analysis
2. merge scripts/fit_model.py structure into nems_db.wrappers
2/3. loader/pre-keyword from modelname maps to a sequence of xforms
4. mulitple xforms for plotting?
5. xforms needed: 
    a. figure out what recording to load and load it
    b. est/val sep
    c. average and compress across stim
    d. pre-fit + fit
    e. plot, save to file and db
6. xforms needed: 
    a. figure out what recording to load and load it
    b. compute state signal
    c. jackkinfe acros interleaved  epochs - est/val breakdown
    d. compute psth
    e. pre-fit + fit
    f. plot, save to file and db

  

## 1. container/load/save for modelspec + associated data ##

Equivalent of load_model and the stack object from the old days. 
Goal: simple way to save a modelspec with all associated recordings and 
metadata so that plots and simulated responses can be regenerated.

syntax:  modelcontainer=load_model(filename)

or  modelspec,est,val = load_model(filename)

STATUS: Almost there. modelname = <loader>_modelspec_<fitter>.  The first keyword
should provide enough information to load the est,val signals. fitter is 
preserved just for bookeeping

PROBLEM: Where to perform est,val breakdown and signal averaging, eg, across
all occurences of each stim. signal averaging could take place on the baphy
server end, but this takes it out of widespread usage.  Could make it a module,
but that will require making modules more flexible so that they can split 
recordings-- right now they just output a new signal.


## 2. wrapper for fitting a model to a single cell in a batch. ##

This is the equivalent of the old fit_single_model.py. Currently a skeleton 
exists for this is nems.utilities.wrappers

syntax: fit_model_baphy(cellid,batch,modelname, **kwargs)

cellid and batch have clear links to celldb structures.

modelname should encode three things:
    1. loading + preprocessing
    2. modelspec
    3. fit routine
    
SVD idea: make keywords for 1. and 3. chop out the middle part fo modelname and
use that to generate the modelspec as in demo_script.

Things that don't appear to be done yet:

1. Evaluation code at the end so that we can populate r_fit,
r_test, r_floor etc. ALMOST DONE. NEED TO COMPUTE r_floor, r_ceiling

2. Save to NarfResults (maybe rename it NemsResults? But I don't
see any reason to change the architecture on the celldb end). MOSTLY DONE. 
NOT ALL METADATA ARE SAVING

3. Save figure somewhere that can be visualized using nems_web. DONE, SAVING
IN SAME PLACE AS MODEL SPEC


## 3. integration with nems_web ##

1. queue up cell/batch/modelname fits to be run by fit_model_baphy
2. pull out fit results and summary plots for display in browser
3. eventually: model inspector--maybe not part of the website

MORE OR LESS BACK TO WHERE THINGS WERE

## 4. more complete summary plot ##

System for generating summary plot that displays something for an arbitrary 
number of modules, have a set of different plot routines associated with 
each module. Also include prediciton correlation results.

## 5. Use case: Fit LN STRF to NAT data ##

batches 271,272,291

Preprocess by averaging across all reps of each stimulus, separate out
high-repeat stimuli for validation

Generate model spec with wcgNN + firNN + dexp1

Pre-fit linear filter without dexp1

Fit

Plot results, save to db

BASICALLY DONE!


## 5a. Use case: Fit LN STRF to SPN data ##

batch 259. Stimulus is already 2-channel envelope. No need for wcNN

Preprocess by averaging across all reps of each stimulus, separate out
high-repeat stimuli for validation

Generate model spec with firNN + dexp1

Pre-fit linear filter without dexp1

Fit

Plot results, save to db

DONE EXCEPT PLOTTING


## 6. Use case: PSTH prediciton + single-trial state ##

Preprocess by generating average response to each stimulus and identifying 
relevant state variables (pupil, task condition, correct/incorrect trial, 
etc.).

Model spec= psthpred_stategain

Fit using n-fold cross-validation - no est/val breakout in the beginning. Fit
n models wiht (n-1)/n of the data, predict the remaining 1/n, generate a 
prediction that concatenates the n sets into a single full prediction.

STATUS: psth, pup, stategain keywords have been generated. Need test data set.
Use VOC? PTD?

STATUS 2: stategain2 is working on NAT + pupil STRF. Need to add state 
permutation for proper controls


## 7. STP support ##

module to execute STP transformation

associated plot routines

can be inserted into LN model


## 8. Miscellaneous things (mostly for SVD to work out) ##

Streamline baphy-NEMS link for various batches, remove the need for
saving the input signals? What to store in NarfData? -- is just the relevant
list of parmfiles enough

More efficient way of saving signals?

Make sure baphy loading works for SPN, SSA, TOR--- appropriate use of env()
and parm stimulus formats.

Multi-channel data? different signals for pop data? Or always save all the
cells from one recording in a single recording???  IDEA: If "cellid" is actually
siteid, then load all cells from that site. How to integrate with batches?
Or should we just get rid of batches and go based on rawids???




