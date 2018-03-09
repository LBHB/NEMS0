import random

def use_all_data(data):
    '''
    Returns a segmentor function (see fitter.py and docs/fitters.md).
    A pass-thru function. Use this when you want to fit on all of your
    estimation set data, all of the time.
    '''
    return data

def random_jackknife_maker(nsplits=10, rebuild_every=50, 
                           invert=False, excise=False):
    '''
    Returns a segmentor function (see fitter.py and docs/fitters.md).

    This segmentor takes a random jackknife of the (estimation set) data
    to fit on for rebuild_every evaluations of the cost function.
    The optional argument nsplits determines how many pieces to split
    the data into before picking
    one of the pieces at random. The optional argument rebuild_every
    decides how many calls to mylambda before the subset is rebuilt.
    '''
    iteration = 0
    subset = None

    def mylambda(data):
        nonlocal iteration
        nonlocal subset
        if (iteration % rebuild_every) == 0 or not subset:
            subset = data.jackknife_by_time(nsplits, random.randint(0, nsplits-1),
                                            invert=invert, excise=True)
        iteration += 1
        return subset

    return mylambda

