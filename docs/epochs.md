# Epochs

## What's an Epoch?

An epoch is an event or period of time that has been tagged with a name. Epochs are attached to Signal objects. A Recording, which contains a collection of signals, has methods that allow us to manipulate the entire collection based on a superset of the epochs in each signal.

An epoch has three values:

	start, end, name

where `start` and `end` denote the beginning and end of the epoch in seconds and `name` is a descriptive string (see [how to name epochs](## epoch-naming). For events (single points in time), the `start` will contain the timestamp of the event and `end` will be NaN. Right now support for events are limited (or nonexistent). As use-cases arise, we will implement support for these.

Epochs are stored in [Signal Objects](signals.md) under the `.epochs` attribute in a pandas DataFrame containing three columns (`start`, `end` and `name`).

For the rest of this document, let's consider some example epoch data in which trials are 40-100 msec long, and there are several TORCs of 20 msec duration played before a tone-in-TORC detection task. In this task, the subject is asked to detect when a tone is played in a sequence of background tokens (known as [TORCs](http://todo/link-to-paper)).

	start           end  	        name

	0.00  		0.060   	ExperimentalTrial
	0.00  		0.05  		PreStimSilence
	0.00  		0.020  		TORC_3983
	0.00  		0.020  		Reference
	0.015   	0.020  		PostStimSilence
	0.020   	0.025  		PreStimSilence
	0.020   	0.040  		TORC_572
	0.020   	0.040  		Reference
	0.035   	0.040  		PostStimSilence
	0.040   	0.045  		PreStimSilence
	0.040   	0.060  		TORC_3983
	0.040   	0.060  		PureTone12
	0.040   	0.060  		DetectionTask
	0.049   	0.049  		Licking
	0.055   	0.060  		PostStimSilence
	0.060   	0.0100  	ExperimentalTrial
	0.060   	0.065  		PreStimSilence
	0.060   	0.080  		TORC_444
	0.060   	0.080  		Reference
	0.066   	0.087  		Licking
	0.075   	0.080  		PostStimSilence
	0.080   	0.0100  	TimeOut

In the example above, note that some epochs have the same name, such as "ExperimentalTrial". If the same name appears several times, we call each appearance an /occurrance/ of a given epoch.

Note also that epochs may overlap. For example, compare the first `TORC_3983` with the first `Reference`. This is a way of indicating that `TORC_3983` is a `Reference` token. This approach facilitates analysis where one may wish to select all reference TORCs and compare them to all TORCs that occur simultaneously with a pure tone (compare the second occurence of `TORC_3983` with `PureTone12`).

This set of epochs tells us quite a bit about what's going on in the experiment. In the first trial two reference TORCs are played (`TORC_3983`, `TORC_572`), then a tone-in-TORC is played (`TORC_3983`, `PureTone12`), and the animal correctly licks. In the second trial, the animal licks during the reference sound and gets a time out.


## How signals use epochs

A [signal object](signal.md) can use epochs to perform two basic operations:

* Mask regions of data. For example, perhaps licking introduces EMG artifacts in to the LFP recordings. In this case, you may want to mask all regions in the LFP recording during a lick so that your analysis isn't affected by these artifacts:

  	 masked_signal = signal.mask_epoch('Licking')

  As you will see later, this masking can also be used to generate subsets of data for cross-validation when fitting models. Signals also have a `select_epochs` method, which is the inverse of `mask_epochs`:

  	 selected_signal = signal.select_epoch('Reference')

* Extract regions of data. For example, perhaps you want to plot the average response to a particular epoch:

     torc = signal.extract_epoch('TORC_3983')
     average_torc = np.nanmean(torc, axis=0)

## Epoch manipulation

Signal objects offer the following methods:

* Getting boundaries of an epoch stored inside the signal using `signal.get_epoch_bounds(epoch_name)`. This will return a Nx2 array (where N is the number of occurances, the first column is start time and the second column is end time).

* Adding epochs to the ones stored inside the signal. You can do this using `signal.add_epoch(epoch_name, epochs)`.

Fancy manipulation of epochs (e.g., selecting epochs that contain another epoch, resizing epoch boundaries, computing the union of two epochs, etc.) must be done outside the signal object. You can then add the newly-created epochs back to the signal object.

### General epoch manipulation

Internally, signal objects store epochs in a DataFrame with three columns ('start', 'end', 'name'). However, when working with epochs outside of the signal object, the epochs will be a 2D array of shape Nx2 (where N is the number of occurences of that epoch, the first column is start time and second column is end time).  In the example below, we have four occurances of the epoch, with the last epoch running from 300 to 301 msec:

    [[0.049  0.049],
     [0.066  0.087],
     [0.145  0.257],
     [0.300  0.301]]

To pull some epochs out for processing, you can use `signal.get_epoch_bounds`:

    dt_epoch = signal.get_epoch_bounds('DetectionTask')
	l_epoch = signal.get_epoch_bounds('Licking')

If we want to take only the correc trials (defined as when the animal licks
during a detection task):

	from nems.data.epochs import epoch_contain
	correct_epoch = epoch_contain(dt_epoch, l_epoch, mode='start')

Then, we can finally do (to NaN everything but the correct epochs):

	masked_signal = signal.select_epoch(correct_epochs)

Great! You can save that for later by adding it to the epochs in the Signal:

	signal.add_epoch('CorrectTrial', correct_epochs)

Then anytime afterward we can simply do:

	correct_signal = signal.select_epoch('CorrectTrial')

### Manipulating epoch boundaries

You can use set theory to manipulate epoch boundaries by subtracting or adding one epoch to the other:

	from nems.data.epochs import epoch_intersection, epoch_difference

	ct_epoch = signal.get_epoch_bounds('CorrectTrial')
	prestim_epoch = signal.get_epoch_bounds('PreStimSilence')

	# Get only the prestim silence by combining using an intersection operation
	only_prestim = epoch_intersection(ct_epoch, prestim_epoch)

	# Remove the prestim silence by using a difference operation
	no_prestim = epoch_difference(ct_epoch, prestim_epoch)


### How do I get the average response to a particular epoch?

Instead of masking data with `signal.select_epoch()` and
`signal.mask_epoch()`, you may also extract epochs:

	data = signal.extract_epoch('TORC_3983')
	average_response = np.nanmean(data, axis=0)

Here, `extract_epoch` returns a 3D array with the first axis containing each occurence of `TORC_3983`. The remaining two axes are channels and time. In this particular situation, the durations of each occurence of `TORC_3983` are identical. However, in some situations, the duration of epochs may vary from occurence to occurence. In this case, shorter epochs will be padded with NaN values so the length matches the longest occurence. To get the average, use `np.nanmean`.

### How do I get the average response in prestim vs poststim, regardless of behavior?

This might be useful for identifying a baseline that is altered by behavior.

	signal.select_epochs('PreStimSilence', inplace=True)
	prestim = signal.as_continuous()
	prestim_mean = np.nanmean(prestim)

	signal.select_epochs('PostStimSilence', inplace=True)
	poststim = signal.as_continuous()
	poststim_mean = np.nanmean(poststim)

### How do I get the average stimulus 300ms before every mistaken lick?

What if we want to know what the animal heard just before it licked accidentally? Or if the TORC was maybe too close to the reference tone?

	# Pull out the epoch we want to analyze
	trial_epoch = signal.get_epoch_bounds('Trials')
	ct_epoch = signal.get_epoch_bounds('CorrectTrials')

	# Note the invert=True. This means to return all trial_epoch that do not
	# contain a ct_epoch.
	bad_trials = epoch_contain(trial_epoch, ct_epoch, invert=True)

	# Extend the 'licking' events backward 300ms
	lick_epoch = signal.get_epoch_bounds('Licking')
	prior_to_licking = adjust_epoch(lick_epoch, -300, 0)

	# Now take the intersection of those two selections
    before_bad_licks = epoch_intersection(bad_trials, prior_to_licking)

	signal.select_epoch(before_bad_licks, inplace=True)
	data = signal.as_continous()
	some_plot_function(data)

Note that `extract_epoch` may end up duplicating data. For example, if the animal licked 10 times a second and you were looking at the 3 seconds prior to each lick, your data will overlap, meaning you just duplicated your total data about 1/2 * 3 * 10 = 15 times! This may negatively alter certain computations of the mean in some sense, and in such circumstances, you may want to use the argument `allow_data_duplication=False` for `signal.extract_epoch()`.

### How do I use epoch info from two different signals in the same recording?

Like signal objects, recording objects offer `mask_epoch` and `extract_epoch` methods. However, you still need to combine the epochs manually. In the above examples, we assumed that a single signal will contain information about both the stimulus and whether the animal licked or not. However, that may not always be the case. Perhaps the "stimulus" signal will contain information about the stimulus and trials while the "lick" signal will contain information about the lick epochs (i.e., how the animal responded). For example, if we want to find anytim the animal blinked or licked and treat those as artifacts and mask the full recording when they occured).

    # The recording version of `get_epoch_bounds` takes the signal name as the
    # first argument and epoch name as the second argument.
	lick_epoch = recording.get_epoch_bounds('lick', 'Licking')
	blink_epoch = recording.get_epoch_bounds('pupil', 'blinks')

	all_artifacts = epochs_union(blink_epoch, lick_epoch)
	recording.mask_signals(all_artifacts)

## Epoch naming

Be descriptive. If you give a stimulus a unique name, then when it occurs in other Recordings,  you can simply concatenate the two recordings and still select exactly the same data.

Avoid implicit indexes like `trial1`, `trial2`, `trial3`; prefer using just `trial` and the folding functionality of `.fold_by('trial')`, which gives you a matrix. If you have truly different stimuli, you may named them `stim01`, `stim02`, but descriptive names like `cookoo_bird.wav`, and `train_horn.wav` are better.

Remember that the idea behind epochs is to tag the content of data, much like HTML marks up text to tell what it is. It's totally fine to tag the exact same epoch with multiple names, if that will help you perform queries on it later.

## What happens with zero-length epochs?

Zero-length epochs are events. They work best with `epochs_contain`:

	trials = signal.get_epochs('Trial')

	# Assume a laser is an event (i.e., a zero-length epoch)
	laser_pulse = signal.get_epochs('Laser')

	laser_trials = epochs_contain(trials, laser_pulse, mode='start')

They will not work with set operations.

## Cross-validation and Jackknifes

	from nems.data.epochs import jacknife_epochs
	stim = recording.get_signal('stim')
	trials = stim.get_epochs('trials')

	# Generate 20 jacknife sets
	jacknifed_trials = jacknife_epochs(n=20)

	results = []
	for jacknife in jacknifed_trials:
		est = recording.mask_epochs(jacknife)
		val = recording.select_epochs(jacknife)
		result = fit_model(est, val, model)
		result.append(result)

	plot_result(result)
	publish_paper(result)
