def ld(loadkey, recording_uri=None, cellid=None):
    '''
    Default loader xfspec. Loads the recording and (optionally) specifies
    input_name and output_name context variables that tell, respectively, the
    name of the input signal fed into the model and output signal to be
    predicted by the model.

    options:
    n : normalize input stim
    revrec : set input_name = 'resp', output_name = 'stim' (ie, reversed from
             default input_name = 'stim, output_name = 'resp')
    popXXX : all cells that don't match cellid should be saved to signal XXX
             default XXX='population'
    p : set input_name = 'psth'

    '''

    options = loadkey.split('.')[1:]

    d = {}
    if recording_uri is not None:
        d['recording_uri_list'] = [recording_uri]
    d['normalize'] = False
    if cellid is not None:
        d['cellid'] = cellid

    for op in options:
        if op.startswith('pop'):
            if len(op) == 3:
                d['save_other_cells_to_state'] = 'population'
            else:
                d['save_other_cells_to_state'] = op[3:]
        elif op == 'revrec':
            d['input_name'] = 'resp'
            d['output_name'] = 'stim'
        elif op == 'p':
            d['input_name'] = 'psth'
        elif op == 'k':
            d['save_other_cells_to_state'] = 'keep'
        elif op == 'n':
            d['normalize'] = True

    xfspec = [['nems.xforms.load_recordings', d]]

    return xfspec
