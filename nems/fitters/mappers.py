"""
Defines mapping functions that return a packer and unpacker (in that order)
corresponding to the type of sigma expected by a fitter.

Our proposed naming convention is to use whatever the modelspec is being
turned into as the name for the mapper function.
"""

from .util import phi_to_vector, vector_to_phi


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
