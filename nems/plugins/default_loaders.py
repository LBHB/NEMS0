def ld(loadkey, recording_uri=None, cellid=None):
    '''Default loader xfspec. Loads the recording, does nothing else.'''

    options = loadkey.split('.')[1:]
    normalize = ('n' in options)

    d = {}
    if recording_uri is not None:
        d['recording_uri_list'] = [recording_uri]
    d['normalize'] = normalize
    if cellid is not None:
        d['cellid'] = cellid

    xfspec = [['nems.xforms.load_recordings', d]]

    return xfspec
