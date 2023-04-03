# update modelspec per bburan model in slack

# take in a modelspec with dists, return sample of points from that dist as phi

# expected value funtion (modelspec -> return mean or w/e)

import scipy.stats as stat
import numpy as np
import nems0.distributions.normal as normal
import nems0.distributions.half_normal as half_normal
# workaround for now. best way to get references to the distribution classes?
# Might be easiest to just rename the distributions in modelpec
# (i.e. 'Normal' --> 'normal.Normal') so that they can be
# getattr'd from nems0.distributions.
dists = {'Normal': normal.Normal,
         'HalfNormal': half_normal.HalfNormal}

def sample_modelspec(modelspec, n_samples):

    samples = {}
    for mod in modelspec:
        phi = mod['phi']
        fn_name = mod['fn']
        for param in list(phi.keys()):
            # Grab tuple representing module's prior distribution.
            #   [0]: type of distribution and
            #   [1]: info about shape of distribution.
            distribution = phi[param]['prior']
            # TODO: How to handle multivariate distribution? Need a separate
            #       distribution class for that? Bit clunky to index into
            #       tuple, then check on length of the params in that index etc
            # TODO: Specified distribution not defined? What to do with error?
            dist_class = dists[distribution[0]](**distribution[1])
            samples[fn_name] = [
                    dist_class.sample()
                    for i in range(n_samples)
                    ]

    return samples

def expected_to_list(modelspec, expectation=None):

    phi_list = []
    for mod in modelspec:
        phi = mod['phi']
        for param in list(phi.keys()):
            distribution = phi[param]['prior']  #[0]: type, [1]: shape
            dist_class = dists[distribution[0]](**distribution[1])
            ex = None

            if not expectation:
                # If not specified, use whatever default initial value
                # is in the modelspec.
                ex = phi[param]['initial']
            elif expectation == 'mean':
                ex = dist_class.mean()
            #elif expected == '...':
            #    TODO: Any others? Should keep this list pretty short,
            #          If other specific values are wanted fitter or w/e
            #          can use a custom function
            #    pass

            phi_list.append(ex)

    return phi_list


def expected_to_dict(modelspec, expectation=None):

    phi_dict = {}
    for mod in modelspec:
        phi = mod['phi']
        fn = mod['fn']
        phi_dict[fn] = {}
        for param in phi.keys():
            distribution = phi[param]['prior']  #[0]: type, [1]: shape
            dist_class = dists[distribution[0]](**distribution[1])
            ex = None

            if not expectation:
                ex = phi[param]['initial']
            elif expectation == 'mean':
                ex = dist_class.mean()
            #elif expected == '...':
            #   TODO: see expected_to_list

            # TODO: better way to do this? Slightly different
            #       from previous phi structure, top level is dict
            #       instead of list but otherwise the same.
            phi_dict[fn][param] = ex

    return phi_dict

modelspec = [#{'fn': 'nems.modules.weight_channels'}
             {'fn': 'nems.modules.fir',       # The pure function to call
              'phi': {                        # The parameters that may change
                  'mu': {
                    # expected distribution
                    'prior': ('Normal', {'mu': 0, 'sd': 1}),
                    # fitted distribution (if applicable)
                    'posterior': None,
                    # initial scalar value, typically the mean of the prior
                    'initial': 0,
                    # fitted scalar vaslue (if applicable)
                    'final': None,
                    },
                  'sd': {
                    'prior': ('HalfNormal', {'sd': 1}),
                    'posterior': None,
                    'initial': 1,
                    'final': None,
                    }
                  },
              'plotfn': 'nems.plots.plot_strf'}, # A plotting function
             ]

samples = sample_modelspec(modelspec, 10)
print(samples)
output_list_none = expected_to_list(modelspec)
print(output_list_none)
output_list_mean = expected_to_list(modelspec, expectation='mean')
print(output_list_mean)
output_dict_none = expected_to_dict(modelspec)
print(output_dict_none)
output_dict_mean = expected_to_dict(modelspec, expectation='mean')
print(output_dict_mean)