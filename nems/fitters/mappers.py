"""
Defines mapping functions that return a packer and unpacker (in that order)
corresponding to the type of sigma expected by a fitter.

Our proposed naming convention is to use whatever the modelspec is being
turned into as the name for the mapper function.
"""

from copy import deepcopy

import numpy as np

from nems.fitters.util import phi_to_vector, vector_to_phi


def simple_vector(modelspec, subset=None):
    """
    Given modelspec, provides functions for converting `phi` to/from a vector

    Parameters
    ----------
    modelspec : list of dictionaries
        Modelspec definition to manipulate
    subset : {None, list of integers}
        If none, pack/unpack entire modelspec otherwise pack/unpack `phi` only
        for specified subset.

    Returns
    -------
    packer : function
        Function with no arguments returning a 1D array, `phi`, that can be used
        by scipy fitters.
    unpacker : function
        Function that takes a 1D array and unpacks it into the original
        modelspec. A modelspec is returned, but this will be the same instance
        as the modelspec provided to `simple_vector` (i.e., it modifies the
        modelspec in-place).

    Note
    ----
    For efficiency reasons, this modifies the modelspec in place (it takes about
    1ms to unpack if we return a new copy, only 25usec if we modify in-place).
    For the `unpacker`, the return value is the same modelspec as the one passed
    into the function (with phi adjusted accordingly).
    """
    if subset is None:
        # Set subset to the full model if not provided
        subset = np.arange(len(modelspec))

    # Create a phi_template only once
    modelspec_subset = [m for i, m in enumerate(modelspec) if i in subset]
    phi_template = [m['phi'] for m in modelspec_subset]

    def packer(modelspec):
        ''' Converts a modelspec to a vector. '''
        nonlocal modelspec_subset

        phi = [m['phi'] for m in modelspec_subset]
        return phi_to_vector(phi)

    def unpacker(vec):
        ''' Converts a vector back into a modelspec. '''
        nonlocal phi_template
        nonlocal modelspec
        nonlocal modelspec_subset

        phi = vector_to_phi(vec, phi_template)
        for i, p in enumerate(phi):
            modelspec_subset[i]['phi'] = p

        return modelspec

    return packer, unpacker


def bounds_vector(modelspec):
    '''
    Converts module bounds from a list of dictionaries to a flattened
    vector.

    Bounds are expected to be defined in the modelspec along the lines of:
        {'fn': '...',
         'phi': {'one': 'value', 'two': 'some other value', 'three': 'test'
                 'four': [0, 0, 0, 0], 'five': [1, 1, 1]
                 'six': [[1, 2, 3],[4, 5, 6]]},
         'bounds': {'one': (-1.0, 1.0), 'two': (0.0, None),
                    'three': (None, None),
                    'four': (None, [0.0, 0.1, 0.2, 0.3]),
                    'five': ([1,2,3], [4,5,6])
                    'six': ([[0,0,0],[1,1,1]], [[2,2,2],[3,3,3]]}}

    Note that each bound is a tuple of the form (lower_bound, upper_bound),
    and a value of None is equivalent to negative or positive infinity.
    The key specifying each bound must also correspond to a key in that
    module's phi dictionary, though there does not need to be a bound
    specified for every entry in phi. Bounds for array-like phis may
    also be defined as a tuple of arrays. The arrays can either be in the
    same shape as the parameter or flattened. Scalar or None bounds for
    array parameters will be broadcast to the size of the array.
    For example:
        ([[0,0,0],[0,0,0]], [[2,2,2],[2,2,2]]),
        ([0,0,0,0,0,0], [2,2,2,2,2,2]),
        (0, 2)
    Would all set equivalent bounds for a 2x3 parameter.
    '''

    phi = [m.get('phi') for m in modelspec]
    phi_vector = phi_to_vector(phi)
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
                        if np.array(this_bound[0]).shape != np.array(v).shape:
                            raise ValueError("Shape of bounds array and "
                                             "phi array does not match for "
                                             "module %d" % i)
                        lowers = [x for x in
                                  np.asanyarray(this_bound[0]).ravel()]
                    if this_bound[1] is None or np.isscalar(this_bound[1]):
                        uppers = [this_bound[1]]*flattened.size
                    else:
                        if np.array(this_bound[1]).shape != np.array(v).shape:
                            raise ValueError("Shape of bounds array and "
                                             "phi array does not match for "
                                             "module %d" % i)
                        uppers = [y for y in
                                  np.asanyarray(this_bound[1]).ravel()]
                    bounds.extend(zip(lowers, uppers))
    n = len(bounds)
    m = len(phi_vector)
    if n != m:
        raise ValueError("Length of bounds vector: %d\n"
                         "Does not match length of phi vector: %d.\n" %
                         (n, m))

    return bounds
