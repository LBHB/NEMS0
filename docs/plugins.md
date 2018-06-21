# PLUGINS

## What are they?

Plugins are an attempt to allow a user or group of users to
either overwrite or add to the modulespec and xforms keywords
found in the NEMS repository without needing to manage different
git branches, deal with conflicting code, or even include
the new code in any sort of version control system at all.

In short, The plugins system imports functions from an arbitrary
number of user-specified paths (that need not be included in
the pythonpath prior to runtime), and adds those functions to the
same registry used by the default NEMS functions.

## Why should you bother using them?

If you have some new keywords or xforms loaders/fitters
that you want to use with NEMS, but don't think they belong
in the main NEMS repository, using plugins can be a lot
simpler than managing git branches.

## How to

Let's say you don't like how the modulespec keyword 'wc' works.
Maybe you need to be able to pass it more options, or maybe you
want to test out some new priors. To do this:

* Create a module where you want the new verison of the function to live.

```
touch /my/custom/code/new_fir.py
```

* Define the function just like any other!

But note that *all* functions
in the module will be registered by default, unless they are preceeded
with one or more underscores or are in all-caps
(to exclude private and global variables, respectively).
```
def wc(kw):
	# Do whatever you want your code to do!
	# Then return the new modulespec structure.
	return {'fn': 'nems.modules.weight_channels',
			'phi': ...}
```   

Alternatively, if you want to use your own code but don't want to
overwrite the core version (for a side-by-side comparison, for example),
you can call your new function whatever you want - just use that new
keyword when defining your modelname.
```
def my_new_wc(kw):
	return {'fn': ...}
```

Note that this is also an easy way to point keywords to a different
implementation of the underlying module by simply changing the
'fn' entry of the returned modulespec.


* Tell nems.config where to find the module

In nems/configs/settings.py (you may need to create settings.py if you
haven't run NEMS yet on a new install), add the path to the new module
to the list for the appropriate plugins variable (or you may need to
start a new list if you haven't set one up yet). For example:
```
KEYWORD_PLUGINS = ['~/jacob/nems_keywords/wc.py']
```

Or, depending on how you prefer to organize your code, you can also just
point to the directory containing the module (which might have other
modules containing other plugins definitions):
```
KEYWORD_PLUGINS = ['~/jacob/nems_keywords/']
```

* You're done!