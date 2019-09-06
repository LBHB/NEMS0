#!/usr/bin/python3

import nems.uri
import nems.xforms as xforms

# Rough test of fitting all recordings in a uri:

recordings_dir = 'http://hyrax.ohsu.edu:3000/recordings/batch271_fs100_ozgf18/'

recording_uris = nems.uri.list_targz_in_nginx_dir(recordings_dir)

keywordstring = 'wc18x1_lvl1_fir15x1_dexp1'

dest = 'http://hyrax.ohsu.edu:3000/results/'

# TODO: Avoid cut and paste copying (this is from fit_model.py)
# We should put this somewhere that many scripts can reuse it.
def fit_model(recording_uri, modelstring, destination):
    '''
    Fit a single model and save it to nems_db.
    '''
    recordings = [recording_uri]

    xfspec = [
        ['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
        ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                         'new_signalname': 'resp',
                                         'epoch_regex': '^STIM_'}],
        ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
        ['nems.xforms.init_from_keywords', {'keywordstring': modelstring}],
        #['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',  {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        ['nems.xforms.plot_summary',    {}]
    ]

    ctx, log = xforms.evaluate(xfspec)

    xforms.save_analysis(destination,
                         recording=ctx['rec'],
                         modelspecs=ctx['modelspecs'],
                         xfspec=xfspec,
                         figures=ctx['figures'],
                         log=log)


for uri in recording_uris:
    fit_model(uri, keywordstring, dest)
