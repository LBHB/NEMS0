class Model:

    def __init__(self):
        self.modules = []

    def append(self, module):
        self.modules.append(module)

    def get_priors(self, data):
        # Here, we query each module for it's priors. A prior will be a
        # distribution, so we set phi to the mean (i.e., expected value) for
        # each parameter, then ask the module to evaluate the input data. By
        # doing so, we give each module an opportunity to perform sensible data
        # transforms that allow the following module to initialize its priors as
        # it sees fit.
        result = data.copy()
        priors = []
        for module in self.modules:
            module_priors = module.get_priors(result)
            priors.append(module_priors)

            phi = {k: p.mean() for k, p in module_priors.items()}
            module_output = module.evaluate(result, phi)
            result.update(module_output)

        return priors

    def evaluate(self, data, phi):
        '''
        Evaluate the Model on some data using the provided modelspec.
        '''
        result = data.copy()
        for module, module_phi in zip(self.modules, phi):
            module_output = module.evaluate(result, module_phi)
            result.update(module_output)

        # We're just returning the final output (More memory efficient. If we
        # get into partial evaluation of a subset of the stack, then we will
        # need to figure out a way to properly cache the results of unchanged
        # parameters such as using joblib).
        return result

    def generate_tensor(self, data, phi):
        '''
        Evaluate the module given the input data and phi

        Parameters
        ----------
        data : dictionary of arrays and/or tensors
        phi : list of dictionaries
            Each entry in the list maps to the corresponding module in the
            model. If a module does not require any input parameters, use a
            blank dictionary. All elements in phi must be scalars, arrays or
            tensors.

        Returns
        -------
        data : dictionary of Signals
            dictionary of arrays and/or tensors
        '''
        # Loop through each module in the stack and transform the data.
        result = data.copy()
        for module, module_phi in zip(self.modules, phi):
            module_output = module.generate_tensor(result, module_phi)
            result.update(module_output)
        return result

    def iget_subset(self, lb=None, ub=None):
        '''
        Return a subset of the model by index
        '''
        model = Model()
        model.modules = self.modules[lb:ub]
        return model

    @property
    def n_modules(self):
        '''
        Number of modules in Model
        '''
        return len(self.modules)
