def ld(loadkey, recording_uri, cellid=None):
    '''Default loader xfspec. Loads the recording, does nothing else.'''
    recordings = [recording_uri]
    options = loadkey.split('.')[1:]
    normalize = ('n' in options)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings,
                'normalize': normalize,
                'cellid': cellid}]]
    return xfspec
