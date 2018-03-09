import numba as nb
import numpy as np
import timeit

from nems.modules import nonlinearity as nl


if __name__ == '__main__':
    np.show_config()
    namespace = {
        #'x': np.arange(-10, 10, 1e-6),
        'base': 1,
        'amplitude': 3,
        'shift': 5,
        'kappa': 0.1,
    }

    n = 25
    kwargs = dict(repeat=100, number=n, globals=namespace, setup='import numpy as np; x=np.arange(-20, 20, 1e-3)')
    x = np.arange(-10, 10, 1e-3)

    namespace['fn'] = nl._double_exponential
    np_result = timeit.repeat('fn(x, base, amplitude, shift, kappa)', **kwargs)

    namespace['fn'] = lambda *args: nl._double_exponential(*args)
    lambda_result = timeit.repeat('fn(x, base, amplitude, shift, kappa)', **kwargs)

    def double_exponential(x, base, amplitude, shift, kappa):
        fn = lambda x: nl._double_exponential(x, base, amplitude, shift, kappa)
        return fn(x)
    namespace['fn'] = double_exponential
    closure_result = timeit.repeat('fn(x, base, amplitude, shift, kappa)', **kwargs)

    namespace['fn'] = nl._double_exponential_ne
    ne_result = timeit.repeat('fn(x, base, amplitude, shift, kappa)', **kwargs)

    namespace['fn'] = nb.jit(nl._double_exponential)
    nb_result = timeit.repeat('fn(x, base, amplitude, shift, kappa)', **kwargs)

    def compare(name, result, reference_result):
        # Always take the *minimum* since this is an indication of how fast the
        # computer can do it. Other values are likely affected by interrupts in
        # other programs.
        result = np.array(result)*1e3/n
        reference_result = np.array(reference_result)*1e3/n
        min_result = np.min(result)
        speedup = np.min(reference_result)/np.min(result)
        print('{}\t{:.4f} msec\tspeedup of {:.4f}x'.format(name, min_result, speedup))

    compare('numpy', np_result, np_result)
    compare('lambda', lambda_result, np_result)
    compare('closure', closure_result, np_result)
    compare('numexpr', ne_result, np_result)
    compare('numba', nb_result, np_result)


