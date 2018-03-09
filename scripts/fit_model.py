#!/usr/bin/python

import sys
import nems.xforms as xforms


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
        # ['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',       {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        ['nems.xforms.plot_summary',    {}],
        # ['nems.xforms.save_recordings', {'recordings': ['est', 'val']}],
    ]

    ctx, log = xforms.evaluate(xfspec)

    xforms.save_analysis(destination,
                         recording=ctx['rec'],
                         modelspecs=ctx['modelspecs'],
                         xfspec=xfspec,
                         figures=ctx['figures'],
                         log=log)



epilog = '''
examples of valid arguments:
  recording      http://potoroo/recordings/TAR010c-02-1.tar.gz
  recording      file:///home/ivar/recordings/
  modelkwstring  wc18x1_lvl1_fir15x1
  modelkwstring  wc18x1_lvl1_fir15x1_dexp1
  destination    http://potoroo/recordings/
  destination    file:///home/ivar/recordings/
 '''

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fit NEMS model',
                                     epilog=epilog,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('recording', help='URI for recording')
    parser.add_argument('modelkwstring', help='Model specification')
    parser.add_argument('destination', help='URI to save result to')
    args = parser.parse_args()
    fit_model(args.recording, args.modelkwstring, args.destination)


if __name__ == '__main__':
    main()
