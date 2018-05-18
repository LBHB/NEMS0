"""
Defines mapping functions that return a packer and unpacker (in that order)
corresponding to the type of sigma expected by a fitter.

Our proposed naming convention is to use whatever the modelspec is being
turned into as the name for the mapper function.
"""
import numpy as np

from nems.fitters.util import phi_to_vector, vector_to_phi


def simple_vector(initial_modelspec):
    """
    Interconverts phi to or from a list of dictionaries to a single
    flattened vector.
    """

    # If you wanted to make a random selection of the parameters,
    # you would do it here, so that the packer and unpacker were both
    # aware of the random subset.

    def packer(modelspec):
        ''' Converts a modelspec to a vector. '''
        phi = [m.get('phi') for m in modelspec]
        vec = phi_to_vector(phi)
        return vec

    def unpacker(vec):
        ''' Converts a vector back into a modelspec. '''
        phi_template = [m.get('phi') for m in initial_modelspec]
        phi = vector_to_phi(vec, phi_template)
        tmp_modelspec = initial_modelspec
        for i, p in enumerate(phi):
            tmp_modelspec[i]['phi'] = p
        return tmp_modelspec

    return packer, unpacker


def bounds_vector(initial_modelspec):
    '''
    Converts module bounds from a list of dictionaries to a flattened
    vector.

    Bounds are expected to be defined in the modelspec along the lines of:
        {'fn': '...',
         'phi': {'one': 'value', 'two': 'some other value', 'three': 'test'},
         'bounds': {'one': (-1.0, 1.0), 'two': (0.0, None),
                    'three': (None, None)}}
    Note that each bound is a tuple of the form (lower_bound, upper_bound),
    and a value of None is equivalent to negative or positive infinity.
    The key specifying each bound must also correspond to a key in that
    module's phi dictionary, though there does not need to be a bound
    specified for every entry in phi.
    '''

    def packer(modelspec):
        phi = [m.get('phi') for m in modelspec]
        bounds = []
        for i, p in enumerate(phi):
            b = modelspec[i].get('bounds', None)
            for k, v in p.items():
                if np.isscalar(v):
                    if b is None:
                        # (None, None) is interpreted as no bounds by fitters
                        bounds.append((None, None))
                    else:
                        bounds.append(b.get(k, (None, None)))
                else:
                    flattened = np.asanyarray(v).ravel()
                    if b is None:
                        bounds.extend([(None, None)]*flattened.size)
                    else:
                        this_bound = b.get(k, (None, None))
                        if this_bound[0] is None or np.isscalar(this_bound[0]):
                            lowers = [this_bound[0]]*flattened.size
                        else:
                            lowers = [x for x in this_bound[0]]
                        if this_bound[1] is None or np.isscalar(this_bound[1]):
                            uppers = [this_bound[1]]*flattened.size
                        else:
                            uppers = [y for y in this_bound[1]]
                        bounds.extend(zip(lowers, uppers))

        return bounds

    def unpacker(vec):
        raise NotImplementedError("No reason to unpack bounds vector, "
                                  "they don't change during fitting.")

    return packer, unpacker
