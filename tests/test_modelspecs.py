import pytest

from nems0.modelspec import get_best_modelspec, sort_modelspecs


@pytest.fixture()
def modelspecs():
    return [[{'fn': 'one',
              'meta': {'r_test': 2.0}}],
            [{'fn': 'two',
              'meta': {'r_test': 3.0}}],
            [{'fn': 'three',
              'meta': {'r_test': 1.0}}]]


def test_sort(modelspecs):
    sort = sort_modelspecs(modelspecs, metakey='r_test', order='ascending')
    fns = [m[0]['fn'] for m in sort]
    assert fns == ['three', 'one', 'two']

    sort = sort_modelspecs(modelspecs, metakey='r_test', order='descending')
    fns = [m[0]['fn'] for m in sort]
    assert fns == ['two', 'one', 'three']


def test_best(modelspecs):
    best = get_best_modelspec(modelspecs, metakey='r_test',
                              comparison='greatest')
    assert best[0][0]['fn'] == 'two'

    best = get_best_modelspec(modelspecs, metakey='r_test',
                              comparison='least')
    assert best[0][0]['fn'] == 'three'
