[Contents](README.md)

# Recording Objects

## What is a recording?

A `Recording` is collection of [signals](signals.md) sampled simultaneously over the same time period. The key point is that all the signals' time indexes line up, which implies that they were sampled at the same rate and have their arrays have the same length in the time dimension.

Recordings provide a way for you to group related data, for example when simultaneously recording neural activity (a signal with several channels, one per neuron), the ambient sound heard (another signal) , and the test subject motion (yet another signal, whose channels describe XYZ coordinates). Because all three signals occurred at the same time and may need to be considered together during analysis, they should be part of the same `Recording`.

## How are Recordings saved?

When saved as files, `Recordings` are represented as directories containing `Signals`, which are represented as 2-3 files. For example, a heircharchy might look lik:

```
└── gus027b13_p_PPS
   ├── gus027b13_p_PPS.pupil.csv
   ├── gus027b13_p_PPS.pupil.json
   ├── gus027b13_p_PPS.resp.csv
   ├── gus027b13_p_PPS.resp.epochs.csv
   ├── gus027b13_p_PPS.resp.json
   ├── gus027b13_p_PPS.stim.csv
   ├── gus027b13_p_PPS.stim.epochs.csv
   └── gus027b13_p_PPS.stim.json 
```

As you can see, 

   1. There is one directory holding all the signals in the recording (`gus027b13_p_PPS` is the name of the recording);

   2. Each signal (`pupil`, `resp`, and `stim` in this case) are represented by two tabular CSV files and one JSON. Files that end in '.epochs.csv' contain information that tag individual events and regions of time. Note that there are no tagged epochs for the `pupil` signal, while the `stim` signal presumably has information tagging when specific sounds were played, and the `resp` signal presumably has information about how the animal behaved or responded.

More details on signal file formats may be found on the [Signals documentation page](signal.md).


## What sort of operations can I do on recordings?

TODO: Describe jackknifing, data splitting for est/val crossvalidation of recordings here. 

