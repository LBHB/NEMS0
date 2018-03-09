import sys
import nems.xforms as xforms
from nems.uri import load_resource


def reload_model(model_uri):
    '''
    Reloads an xspec and modelspec that were saved in some directory somewhere.
    This recreates the context that occurred during the fit.
    Passes additional context {'IsReload': True}, which xforms should react to
    if they are not intended to be run on a reload.
    '''
    xfspec_uri = model_uri + 'xfspec.json'

    # TODO: instead of just reading the first modelspec, read ALL of the modelspecs
    # I'm not sure how to know how many there are without a directory listing!
    modelspec_uri = model_uri + 'modelspec.0000.json'

    xfspec = load_resource(xfspec_uri)
    modelspec = load_resource(modelspec_uri)

    ctx, reloadlog = xforms.evaluate(xfspec, {'IsReload': True,
                                              'modelspecs': [modelspec]})

    return ctx


def print_usage():
    print('''
Usage:
      ./reload_model.py <model_uri>

Examples of valid <model_uri>:
  http://potoroo/results/TAR010c-02-1/wc18x1_lvl1_fir15x1/None/2018-03-07T22%3A55%3A11/
  /home/ivar/results/TAR010c-18-1/wc18x1_lvl1_fir15x1/fit_basic/2018-03-07T23:40:48/
 ''')

# Parse the command line arguments and do the fit
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
    else:
        ctx = reload_model(sys.argv[1])
        print('Successfully reloaded context: {}'.format(ctx))
        # If you need to do something else with that context, you could do it here.
