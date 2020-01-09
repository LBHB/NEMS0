# We get a Theano import error otherwise. Probably best to set this in the
# virtual environment instead.
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pymc3 as mc

from nems.distributions.api import Normal, HalfNormal, Beta, Gamma


prior_map = {
    Normal: lambda n, d: mc.Normal(n, mu=d.mu, sd=d.sd, shape=d.mu.shape),
    HalfNormal: lambda n, d: mc.HalfNormal(n, sd=d.sd, shape=d.sd.shape),
    Beta: lambda n, d: mc.Beta(n, alpha=d.alpha, beta=d.beta, shape=d.alpha.shape),
    Gamma: lambda n, d: mc.Gamma(n, alpha=d.alpha, beta=d.beta, shape=d.alpha.shape),
}


def construct_priors(nems_priors):
    '''
    Convert the NEMS priors to the format required by PyMC3.

    This conversion is pretty straightforward as the attribute names for NEMS
    priors are designed to map directly to the attributes on the corresponding
    PyMC3 prior class. The only reason I don't actually use PyMC3 priors (in
    lieu of our custom class) is to minimize dependencies on third-party
    libraries.  For example, if you don't want to do Bayes fitting or
    variational inference, then you don't need PyMC3.  However, priors are still
    extremely useful for non-Bayes fitters as they provide information about
    constraints (see the scipy package).
    '''
    mc_priors = []
    for module_priors in nems_priors:
        module_mc_priors = {}
        for name, prior in module_priors.items():
            dtype = type(prior)
            converter = prior_map[dtype]
            module_mc_priors[name] = converter(name, prior)
        mc_priors.append(module_mc_priors)
    return mc_priors


def construct_bayes_model(nems_model, signals, pred_name, resp_name,
                          batches=None):
    '''
    Builds the Bayesian version of the NEMS model. This essentially converts the
    NEMS set of modules into a symbolic evaluation graph that is used for
    maximizing the likelihood of a Poisson prior.
    '''
    signals = signals.copy()
    nems_priors = nems_model.get_priors(signals)

    # Now, batch the signal if requested. The get_priors code typically doesn't
    # work with batched tensors, so we need to do this *after* getting the
    # priors.
    if batches is not None:
        for k, v in signals.items():
            signals[k] = mc.Minibatch(v, batch_size=batches)

    with mc.Model() as mc_model:
        mc_priors = construct_priors(nems_priors)
        tensors = nems_model.generate_tensor(signals, mc_priors)
        pred = tensors[pred_name]
        obs = tensors[resp_name]
        likelihood = mc.Poisson('likelihood', mu=pred, observed=obs)
    return mc_model


def trace_to_phi(trace, priors, statistic=np.mean):
    phi = []
    for module_priors in priors:
        module_phi = {}
        for name in module_priors:
            values = trace.get_values(name)
            module_phi[name] = statistic(values, axis=0)
        phi.append(module_phi)
    return phi


def fit(nems_model, signals, n=30000, pred_name='pred', resp_name='resp'):
    '''
    Fit model using automatic differentiation variational inference (ADVI)

    Parameters
    ----------
    nems_model : instance of `nems.Model`
        Model to fit
    signals : dictionary of arrays
        Data to fit
    method : {'advi', 'fullrank_advi'}
        See documentation for `pymc3.fit`.
    n : int
        Number of iterations
    pred_name : string
        Name of prediction in output of the model
    resp_name : string
        Name of response in output of the model

    Note
    ----
    This fitting method gives you an estimate of the confidence interval for
    each parameter. This information can be used to fine-tune the model during
    subsequent steps. ADVI is computationally faster than standard Bayes
    regression, but only gives you an approximation of the posterior (i.e., the
    confidence interval).
    '''
    mc_model = construct_bayes_model(nems_model, signals, pred_name, resp_name)
    with mc_model:
        result = mc.fit(n)
        trace = result.sample(5000)

    priors = nems_model.get_priors(signals)
    return {
        'advi': result,
        'trace': trace,
        'phi': trace_to_phi(trace, priors),
        'phi_sd': trace_to_phi(trace, priors, statistic=np.std),
    }


def fit_bayes(nems_model, signals, n=10000, pred_name='pred', resp_name='resp'):
    '''
    Fit model using Bayes inference

    Parameters
    ----------
    nems_model : instance of `nems.Model`
        Model to fit
    signals : dictionary of arrays
        Data to fit
    pred_name : string
        Name of prediction in output of the model
    resp_name : string
        Name of response in output of the model

    Returns
    -------
    advi :
        This is not directly usable by NEMS as-is.

    Note
    ----
    This fitting method gives you an estimate of the confidence interval for
    each parameter. This information can be used to fine-tune the model during
    subsequent steps. Bayes is computationally slower than ADVI, but gives you
    an accurate measure of the posterior.
    '''
    mc_model = construct_bayes_model(nems_model, signals, pred_name, resp_name)
    with mc_model:
        result = mc.sample(n)
    return result

