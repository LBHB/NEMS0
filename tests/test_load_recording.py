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
import nems.recording as recording
from nems.uri import get_demo_recordings

def test_load_recording():
    get_demo_recordings()

    nems_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
    targz = nems_dir + '/recordings/TAR010c-18-1.tgz'

    rec = recording.load_recording(targz)

