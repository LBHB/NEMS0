[Contents](README.md)

# XFORMS

## What is the problem we trying to solve with XFORMS? What are XFORMS?

Often times, after fitting many different modelspecs and spending thousands of computer hours fitting models, you may find yourself needing to reload some models and the exact data that was used to fit it, for example when generating new plots. There are several ways that you could try to perfectly recreate the environment the model was fit in, and we will focus on two: 1) you could save a snapshot of the script used to fit the model; or 2) you could saving the recordings themselves (which may have been split into estimation, validation, and other sub-recordings). 

XFORMS are essentially option #1. An xform (short for "transformation") is at its essence a python function that takes in a dictionary and spits out a dictionary. We call these dictionaries the "context", and they are a way of keeping track of variables that would otherwise be contained in the script namespace. This trick is easier in some languages, such as Lisps and other homoiconic languages, but still is useful in Python and not too difficult. 

Basically, instead of writing a lines in a script like this:

```
rec = Recording.load("http://potoroo/baphy/271/bbl086b-11-1")

rec = preproc.add_average_sig(rec,
                              signal_to_average='resp',
                              new_signalname='resp',
                              epoch_regex='^STIM_')
```

You instead write a line in a list of xforms in a structure like this:

```
xfspec = [['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
          ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                           'new_signalname': 'resp',
                                           'epoch_regex': '^STIM_'}],
```

Note that the above is trivially serialized to JSON for saving. In 'nems.xforms' package, you capture the implicit variable `rec` in the xforms functions:

```
def load_recordings(recording_uri_list, **context):
    '''
    Load one or more recordings into memory given a list of URIs.
    '''
    rec = Recording.load(recording_uri_list[0])
    other_recordings = [Recording.load(uri) for uri in recording_uri_list[1:]]
    if other_recordings:
        rec.concatenate_recordings(other_recordings)
    return {'rec': rec}


def add_average_sig(rec, signal_to_average, new_signalname, epoch_regex,
                    **context):
    rec = preproc.add_average_sig(rec,
                                  signal_to_average=signal_to_average,
                                  new_signalname=new_signalname,
                                  epoch_regex=epoch_regex)
    return {'rec': rec}
```

It's purely convention, but we advise keeping `nems.xforms` function names short and not doing too much inside each transform. 
