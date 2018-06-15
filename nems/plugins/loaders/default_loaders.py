def load(loadkey, recording_uri):
    '''Default loader xfspec. Loads the recording, does nothing else.'''
    recordings = [recording_uri]
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings, 'normalize': False}]]
    return xfspec
