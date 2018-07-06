def load(loadkey, recording_uri):
    '''Default loader xfspec. Loads the recording, does nothing else.'''
    recordings = [recording_uri]
    options = loadkey.split('.')[1:]
    normalize = ('n' in options)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings, 'normalize': normalize}]]
    return xfspec
