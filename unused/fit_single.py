#!/usr/bin/env python3

# This script runs nems.fit_single_model from the command line

from nems import xform_helper
from nems import xforms
import sys
import os

import logging
log = logging.getLogger(__name__)

        
if __name__ == '__main__':
    
    # leftovers from some industry standard way of parsing inputs
    
    #parser = argparse.ArgumentParser(description='Generetes the topic vector and block of an author')
    #parser.add_argument('action', metavar='ACTION', type=str, nargs=1, help='action')
    #parser.add_argument('updatecount', metavar='COUNT', type=int, nargs=1, help='pubid count')
    #parser.add_argument('offset', metavar='OFFSET', type=int, nargs=1, help='pubid offset')
    #args = parser.parse_args()
    #action=parser.action[0]
    #updatecount=parser.updatecount[0]
    #offset=parser.offset[0]
        
    if len(sys.argv)<3:
        print('Two parameters required.')
        print('Syntax: fit_single <modelname> <recording_uri>')
        exit(-1)

    modelname=sys.argv[1]
    recording_uri=sys.argv[2]
    
    log.info("Running fit_single(%s, %s)", modelname,recording_uri)
    
    meta = {'cellid': recording_uri, 'modelname': modelname,
        'githash': os.environ.get('CODEHASH', ''),
        'recording_uri': recording_uri}

    # set up sequence of events for fitting
    xfspec = xform_helper.generate_xforms_spec(recording_uri, modelname, meta=meta)
    
    # actually do the fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save results
    destination = os.path.dirname(recording_uri)
    log.info('Saving modelspec(s) to %s ...', destination)
    save_data = xforms.save_analysis(destination,
                                     recording=ctx['rec'],
                                     modelspecs=ctx['modelspecs'],
                                     xfspec=xfspec,
                                     figures=ctx['figures'],
                                     log=log_xf)
    savepath = save_data['savepath']

    log.info('Done.')
           
