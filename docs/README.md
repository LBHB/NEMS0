## NEMS - Major elements of fit & analysis process

### Load data

Models are fit using datasets packaged as [recordings](recordings.md), which are a collection of [signals](signals.md), typically at least a stimulus and response.

1. Get some demo data with `nems.recording.get_demo_data`

2. Load recording from file saved in NEMS native format. `nems.recording.load_recording`

3. Generate a recording on the fly, e.g., with `nems.recording.load_recording_from_arrays`

### Preprocess the data

1. Mask out "bad" epochs

2. Generate state signals from epochs

3. Transform the stimulus(?)

### Define a modelspec

The [modelspec](modelspecs.md) defines a sequence of transformations. Each transformation is called a [module](modules.md), and applies a transformation to a signal that models some stage of neural processing. Modules include linear reweighting, FIR filter, static nonlinearity, etc. To define the modelspec:

1. Use keywords

    * `modelspec=nems.initializers.from_keywords(model_keyword_string)`

    * default keywords in `nems.plugins.default_keywords`

    * see [keywords.md]()

2. Assemble the modelspec as a list of modules

### Perform the model fit

1. Load parameters from a simpler, previously fit model

2. Define jack-knife subsets for n-fold cross-validation

3. "Pre-fit" a subset of the model parameters (`nems.initializers`)

4. Fit a state-independent model

5. Fit a fully state dependent model


### Evaluate the model


### Plot the results


### Save results


### Reload for analysis later


