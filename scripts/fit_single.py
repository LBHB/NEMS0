#!/usr/bin/env python3

# This script runs xhelp.fit_model_xform from the command line

import os
import sys
import logging
log = logging.getLogger(__name__)

force_SDB=True
try:
    if sys.argv[3][:4] == 'SDB-':
        force_SDB=True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'

import nems.xform_helper as xhelp
import nems.utils

if force_SDB:
    log.info('Setting OPENBLAS_CORETYPE to sandybridge')

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
        nems.utils.progress_fun = nd.update_job_tick

        if 'SLURM_JOB_ID' in os.environ:
            nd.update_job_pid(os.environ['SLURM_JOB_ID'])

    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    if len(sys.argv) < 4:
        print('syntax: fit_single cellid batch modelname')
        exit(-1)

    cellid = sys.argv[1]
    batch = sys.argv[2]
    modelname = sys.argv[3]

    log.info("Running xform_helper.fit_model_xform({0},{1},{2})".format(cellid, batch, modelname))
    #savefile = nw.fit_model_xforms_baphy(cellid, batch, modelname, saveInDB=True)
    savefile = xhelp.fit_model_xform(cellid, batch, modelname, saveInDB=True)

    log.info("Done with fit.")

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if db_exists & bool(queueid):
        nd.update_job_complete(queueid)


