def ld(loadkey, recording_uri=None, cellid=None):
    '''Default loader xfspec. Loads the recording, does nothing else.'''
    recordings = [recording_uri]
    options = loadkey.split('.')[1:]
    normalize = ('n' in options)
    cst = ('cst' in options)

    if recording_uri is not None:
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings,
                    'normalize': normalize,
                    'cellid': cellid}]]
    else:
        xfspec = [['nems.xforms.load_recordings',
                   {'normalize': normalize,
                    'cellid': cellid}]]

    return xfspec
