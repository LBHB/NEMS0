import numpy as np

def pup_dep_LV(result, pred_name='pred', resp_name='resp', **context):
    '''
    For purely LV model. Constrain first LV (lv_slow) to correlate with pupil,
    second LV (lv_fast) to have variance that correlates with pupil.
    Weigh these constraints vs. minimizing nsme.

    Will also work if only have one or the other of the two LVs
    '''
    result = result.apply_mask()
    lv_chans = result['lv'].chans
    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()
    respstd = np.nanstd(X2)
    squared_errors = (X1-X2)**2
    mse = np.sqrt(np.nanmean(squared_errors))
    nmse = mse / respstd

    alpha = context['alpha']
    signed_correlation = context.get('signed_correlation', False)

    if len(lv_chans) > 3:
        raise ValueError("Not set up to handle greater than 2 LVs right now due to \
                        complications with hyperparameter specification")

    if type(alpha) is dict:
        # passed different hyperparameters for each of the LVs
        fast_alpha = alpha['fast_alpha']
        slow_alpha = alpha['slow_alpha']

        if (fast_alpha + slow_alpha) > 1:
                raise ValueError("Hyperparameter values must sum to < 1")

    else:
        fast_alpha = slow_alpha = alpha

    if ('lv_fast' not in lv_chans) & ('lv_slow' not in lv_chans):
        # don't know how to constrain LV(s), just minimizing nmse
        return nmse
    
    elif ('lv_fast' in lv_chans) & ('lv_slow' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        
        fast_lv_chans = [c for c in lv_chans if 'lv_fast' in c]
        fast_cc = []
        p = np.mean(p, axis=-1)
        for c in fast_lv_chans:
            lv_fast = result['lv'].extract_channels([c])._data.reshape(-1, ref_len)
            lv_fast = np.std(lv_fast, axis=-1)
            if signed_correlation:
                cc = lv_corr_pupil(p, lv_fast)
            else:
                cc = -abs(lv_corr_pupil(p, lv_fast))
            fast_cc.append(cc)


        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - (slow_alpha + fast_alpha)) * nmse)

        for i, c in enumerate(fast_lv_chans):
            cost += (fast_alpha * fast_cc[i])

        return cost

    elif ('lv_fast' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        lv_fast = result['lv'].extract_channels(['lv_fast'])._data.reshape(-1, ref_len)
        
        p = np.mean(p, axis=-1)
        lv_fast = np.std(lv_fast, axis=-1)
        if signed_correlation:
            fast_cc = lv_corr_pupil(p, lv_fast)
        else:
            fast_cc = -abs(lv_corr_pupil(p, lv_fast))

        if np.sum(lv_fast)==0:
            fast_cc = 0

        cost = (fast_alpha * fast_cc) + ((1 - fast_alpha) * nmse)
        return cost

    elif ('lv_slow' in lv_chans):
        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - slow_alpha) * nmse)
        return cost

def lv_corr_pupil(p, lv):
    """
    return correlation of pupil and lv
    """
    return np.corrcoef(p.squeeze(), lv.squeeze())[0, 1]