from .signal import merge_selections

class CrossValidator:

    def __init__(self, folds):
        self.folds = folds

    def fit(self, fitter, model, signals):
        for train, test in self.split_train_test(signals):
            train_result = fitter.fit(model, train)
            test_result = model.evaluate(train_result, test)

def cross_validate_wrapper(fitter_fn, n_jackknifes, data, modelspec):
    '''
    Wraps any fitter function with an n-fold cross-validation.
    '''
    get_jackknifed_subset = lambda i: data.jackknifed_by_time(n_jackknifes,
                                                              i)    
    get_held_out_subset = lambda i: data.jackknifed_by_time(n_jackknifes,
                                                            i, invert=True)
    jackknifed_modelspecs = [modelspec] * n_jackknifes
    predictions = [data['resp']] * n_jackknifes
    for i in range(n_jackknifes):
        d = get_jackknifed_subset(i)
        m = jackknifed_modelspecs[i]
        jackknifed_modelspecs[i] = fitter_fn(d, m):
        v = get_held_out_subset(i)
        predictions[i] = evaluator(v, jackknifed_modelspecs[i])
    combined_pred = merge_selections(predictions)
    return combined_pred
