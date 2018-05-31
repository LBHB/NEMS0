#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:26:27 2018

@author: svd
"""

import io
import os
from pathlib import Path
import tarfile
import tempfile
import shutil

from nems.signal import load_signal_from_streams
from nems.recording import Recording


def test_load_recording():
    targz = Path(__file__).parents[1] / 'recordings/resp2.tgz'
    # tgz_stream = open(targz, 'rb')


    tpath=tempfile.mktemp()

    streams = {}  # For holding file streams as we unpack
    with tarfile.open(targz, mode='r:gz') as t:
        for member in t.getmembers():
            if member.size == 0:  # Skip empty files
                continue
            basename = os.path.basename(member.name)
            # Now put it in a subdict so we can find it again
            signame = str(basename.split('.')[0:2])
            if basename.endswith('epoch.csv'):
                keyname = 'epoch_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            elif basename.endswith('.csv'):
                keyname = 'data_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            elif basename.endswith('.h5'):
                keyname = 'data_stream'

                # current non-optimal solution. extract hdf5 file to disk and then load
                t.extract(member,tpath)
                f = tpath+'/'+member.name

            elif basename.endswith('.json'):
                keyname = 'json_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            else:
                m = 'Unexpected file found in tar.gz: {} (size={})'.format(member.name, member.size)
                raise ValueError(m)
            # Ensure that we can doubly nest the streams dict
            if signame not in streams:
                streams[signame] = {}
            # Read out a stringIO object for each file now while it's open
            #f = io.StringIO(t.extractfile(member).read().decode('utf-8'))
            streams[signame][keyname] = f

    # Now that the streams are organized, convert them into signals
    # log.debug({k: streams[k].keys() for k in streams})
    signals = [load_signal_from_streams(**sg) for sg in streams.values()]
    signals_dict = {s.name: s for s in signals}

    rec = Recording(signals=signals_dict)

    shutil.rmtree(tpath) # clean up
