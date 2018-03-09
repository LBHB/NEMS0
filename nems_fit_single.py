#!/usr/bin/env python3

# This script runs nems_main.fit_single_model from the command line

import nems.utilities.wrappers as nw
import sys
import os

import logging
log = logging.getLogger(__name__)

try:
    import nems.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems.db, can't update tQueue")
    print(e)
    db_exists = False
        
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
    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
    else:
        queueid = 0
        
    if queueid:
        print("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    if len(sys.argv)<4:
        print('syntax: nems_fit_single cellid batch modelname')
        exit(-1)

    cellid=sys.argv[1]
    batch=sys.argv[2]
    modelname=sys.argv[3]
    
    print("Running fit_single_model({0},{1},{2})".format(cellid,batch,modelname))
    savefile = nw.fit_model_baphy(cellid,batch,modelname,saveInDB=True)

    print("Done with fit.")
    
    # Edit: added code to save preview image. -Jacob 7/6/2017
    #preview_file = stack.quick_plot_save(mode="png")
    #print("Preview saved to: {0}".format(preview_file))
    
    #if db_exists:
    #    if queueid:
    #        pass
    #    else:
    #        queueid = None
    #    r_id = nd.save_results(stack, preview_file, queueid=queueid)
    #    print("Fit results saved to NarfResults, id={0}".format(r_id))

    # Mark completed in the queue. Note that this should happen last thing! 
    # Otherwise the job might still crash after being marked as complete.
    if queueid:
        nd.update_job_complete(queueid)
        
       
