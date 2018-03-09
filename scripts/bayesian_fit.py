import numpy as np

from nems.signal import load_signals_in_dir
from nems.model import Model
from nems.modules.api import (WeightChannelsGaussian, FIR, DoubleExponential)

from nems.fitters import bayes_fitter as bf


if __name__ == '__main__':
    signals = load_signals_in_dir('signals/zee021e02_p_RDT')

    # Temporary hack to pull arrays out of the Signal object until we decide
    # what to do with the data format
    data = {k: v.__matrix__ for k, v in signals.items()}

    # Strip NaN values since the fitters don't like these.
    for k, v in data.items():
        v[np.isnan(v)] = 0

    # Use the first 80% as estimation data, and the last 20% as validation. Not
    # implemented yet!
    #est, val = split_signals_by_time(signals, 0.8)

    nems_model = Model()

    wcg = WeightChannelsGaussian(2, 'stim2', 'pred1')
    fir = FIR(15, 'pred1', 'pred1')
    dexp = DoubleExponential('pred1', 'pred1', 'resp-c1')

    model.append(wcg)
    model.append(fir)
    model.append(dexp)

    mc_model = bf.construct_bayes_model(model, data, 'pred1', 'resp-c1')

    # Eventually we would wrap this up into a fit function, but for now ...
    with mc_model:
        result = mc.fit()
    trace = result.sample(5000)

    initial_phi = []
    fitted_phi = []
    for module_priors in model.get_priors(data):
        module_initial_phi = {}
        module_fitted_phi = {}
        for name in module_priors:
            module_initial_phi[name] = module_priors[name].mean()
            module_fitted_phi[name] = trace.get_values(name).mean(axis=0)
        initial_phi.append(module_initial_phi)
        fitted_phi.append(module_fitted_phi)

    # Fitter
    #eval_fn = partial(model.evaluate, est)
    #cost_fn = lambda i, o: MSE(i['resp'], o['pred'])
    #fitter = LinearFitter(cost_fn, eval_fn)

    # The time consuming part
    #phi_distributions = fitter.fit(model)

    # Plot the prediction vs reality
    # phi_distributions.plot('/some/other/path.png')

    # TODO: Plot confidence intervals
    # phi_EV = phi_distributions.expected_value()
    # phi_10 = phi_distributions.percentile(10)
    # phi_90 = phi_distributions.percentile(90)
    # pred_EV = model.evaluate(phi_EV, val)
    # pred_10 = model.evaluate(phi_10, val)
    # pred_90 = model.evaluate(phi_90, val)
    # plot_signals('/some/path.png', pred_EV, pred_10, pred_90, ...)

    # TODO: At a later date, cross-validate
    # validator = CrossValidator(fitter, 20)
    # validator.fit(fitter, model, signals)

    # TODO: Measure various other performance metrics and save them
    # performance = {'mse': MSE(val, pred_EV),
    #                'logl': LogLikelihood(val, pred_EV)}
    # model.save('/yet/another/path.json', phi_distributions, performance)
