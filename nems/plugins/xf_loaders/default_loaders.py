def ozgf(loadkey, recording_uri):
    recordings = [recording_uri]

    if loadkey in ["ozgf100ch18", "ozgf100ch18n"]:
        normalize = int(loadkey == "ozgf100ch18n")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems.xforms.average_away_stim_occurrences', {}]]

    elif loadkey in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        normalize = int(loadkey == "ozgf100ch18npup")
        xfspec = [['nems.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems.xforms.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}]]

    return xfspec


def env(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}],
              ['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


def psth(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}],
              ['nems.xforms.generate_psth_from_resp', {}],
              ['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


def psths(loadkey, recording_uri):

    recordings = [recording_uri]
    state_signals, permute_signals = _state_model_loadkey_helper(loadkey)
    xfspec = [['nems.xforms.load_recordings',
               {'recording_uri_list': recordings}],
              ['nems.xforms.generate_psth_from_resp',
               {'smooth_resp': True}],
              ['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


def _state_model_loadkey_helper(loadkey):

    if loadkey.endswith("beh0"):
        state_signals = ['active']
        permute_signals = ['active']
    elif loadkey.endswith("beh"):
        state_signals = ['active']
        permute_signals = []
    elif loadkey.endswith("pup0beh0"):
        state_signals = ['pupil', 'active']
        permute_signals = ['pupil', 'active']
    elif loadkey.endswith("pup0beh"):
        state_signals = ['pupil', 'active']
        permute_signals = ['pupil']
    elif loadkey.endswith("pupbeh0"):
        state_signals = ['pupil', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("pupbeh"):
        state_signals = ['pupil', 'active']
        permute_signals = []

    elif loadkey.endswith("pup0pre0beh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pupil', 'pre_passive']
    elif loadkey.endswith("puppre0beh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pre_passive']
    elif loadkey.endswith("pup0prebeh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = ['pupil']
    elif loadkey.endswith("pupprebeh"):
        state_signals = ['pupil', 'pre_passive', 'active']
        permute_signals = []

    elif loadkey.endswith("pre0beh0"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['pre_passive', 'active']
    elif loadkey.endswith("pre0beh"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['pre_passive']
    elif loadkey.endswith("prebeh0"):
        state_signals = ['pre_passive', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("prebeh"):
        state_signals = ['pre_passive', 'active']
        permute_signals = []

    elif loadkey.endswith("predif0beh"):
        state_signals = ['pre_passive', 'puretone_trials',
                         'hard_trials', 'active']
        permute_signals = ['puretone_trials', 'hard_trials']
    elif loadkey.endswith("predifbeh"):
        state_signals = ['pre_passive', 'puretone_trials',
                         'hard_trials', 'active']
        permute_signals = []
    elif loadkey.endswith("pbs0pev0beh0"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs', 'pupil_ev', 'active']
    elif loadkey.endswith("pbspev0beh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_ev']
    elif loadkey.endswith("pbs0pevbeh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs']
    elif loadkey.endswith("pbspevbeh0"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['pupil_bs', 'pupil_ev']
    elif loadkey.endswith("pbs0pev0beh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = ['active']
    elif loadkey.endswith("pbspevbeh"):
        state_signals = ['pupil_bs', 'pupil_ev', 'active']
        permute_signals = []
    else:
        raise ValueError("invalid loadkey string")

    return state_signals, permute_signals
