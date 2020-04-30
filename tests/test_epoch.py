import pytest

import numpy as np
import pandas as pd

from nems.epoch import (epoch_union, epoch_difference, epoch_intersection,
                        epoch_contains, epoch_contained, adjust_epoch_bounds,
                        remove_overlap, find_common_epochs, add_epoch)

@pytest.fixture()
def epoch_a():
    return np.array([
        [  0,  50],
        [ 60,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
        [140, 150],
     ])


@pytest.fixture()
def epoch_b():
    return np.array([
        [ 55,  70],
        [ 75,  76],
        [ 90,  95],
        [110, 120],
     ])


@pytest.fixture()
def epoch_c():
    return np.array([
        [  0,  70],
        [ 85, 100],
        [ 92, 115],
     ])


@pytest.fixture()
def epoch_df():
    epochs = [
        ['parent',   1, 11],
        ['parent_1', 1, 11],
        ['child_a',  1, 2],
        ['child_b',  1, 6],
        ['child_c',  6, 7],
        ['child_d',  7, 11],
        ['child_e',  9, 11],
        ['parent',   30, 40],
        ['parent_2', 30, 40],
        ['child_a',  30, 31],
        ['child_b',  30, 35],
        ['child_c',  35, 36],
        ['child_d',  36, 40],
    ]
    epochs = pd.DataFrame(epochs, columns=['name', 'start', 'end'])
    return epochs


def test_intersection(epoch_a, epoch_b):
    expected = np.array([
        [ 60,  70],
        [ 75,  76],
        [ 90,  95],
    ])
    result = epoch_intersection(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_intersection_bug():
    # Copied from real data
    a = np.array([
        [57.7, 61.4],
        [61.4, 66.6],
        [66.6, 70.9],
        [70.9, 75.80000000000001],
        [75.8, 81.0],
    ])

    b = np.array([
        [57.7000000000001, 61.4000000000001],
        [61.4000000000001, 66.6000000000001],
        [66.6000000000001, 70.9],
        [75.8000000000001, 81.0000000000001],
        [81.0000000000001, 85.6000000000001],
    ])

    expected = np.array([
        [57.7, 61.4],
        [61.4, 66.6],
        [66.6, 70.9],
        [75.8, 81.0],
    ])

    result = epoch_intersection(a, b)
    print(result)
    assert np.all(result == expected)


def test_intersection_float(epoch_a, epoch_b):
    expected = np.array([
        [ 60,  70],
        [ 75,  76],
        [ 90,  95],
    ])/10
    result = epoch_intersection(epoch_a/10, epoch_b/10)
    assert np.all(result == expected)


# now failing because check removed
@pytest.mark.xfail
def test_empty_intersection():
    a = np.array([
        [  0, 20],
        [ 50, 70]
    ])

    b = np.array([
        [ 25, 30],
        [ 30, 45]
    ])

    with pytest.warns(RuntimeWarning):
        result = epoch_intersection(a, b)
        print(result)


def test_union(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 55,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
        [110, 120],
        [140, 150],
    ])
    result = epoch_union(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_union_float(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 55,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
        [110, 120],
        [140, 150],
    ])/10
    result = epoch_union(epoch_a/10, epoch_b/10)
    assert np.all(result == expected)


def test_difference(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 77,  77],
        [ 85,  90],
        [ 95, 100],
        [140, 150],
    ])
    result = epoch_difference(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_difference_float(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 77,  77],
        [ 85,  90],
        [ 95, 100],
        [140, 150],
    ])/10
    result = epoch_difference(epoch_a/10, epoch_b/10)
    assert np.all(result == expected)


# now failing because check removed
@pytest.mark.xfail
def test_empty_difference():
    a = b = np.array([
        [  0,  50],
        [ 50, 100]
    ])

    with pytest.warns(RuntimeWarning):
        result = epoch_difference(a, b)


def test_contains(epoch_a, epoch_b):
    expected_any = np.array([
        False,
        True,
        True,
        False,
        True,
        False,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'any')
    assert np.all(actual == expected_any)

    expected_start = np.array([
        False,
        False,
        True,
        False,
        True,
        False,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'start')
    assert np.all(actual == expected_start)

    expected_end = np.array([
        False,
        True,
        True,
        False,
        True,
        False,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'end')
    assert np.all(actual == expected_end)

    expected_both = expected_start & expected_end
    actual = epoch_contains(epoch_a, epoch_b, 'both')
    assert np.all(actual == expected_both)


def test_epoch_contained(epoch_a, epoch_b, epoch_c):
    expected = np.array([
        False,
        True,
        True,
        False,
        False,
        False,
    ])
    actual = epoch_contained(epoch_a, epoch_b)
    assert np.all(actual == expected)

    expected = np.array([
        False,
        True,
        True,
        False,
    ])
    actual = epoch_contained(epoch_b, epoch_a)
    assert np.all(actual == expected)

    expected = np.array([
        True,
        True,
        False,
        False,
        True,
        False,
    ])
    actual = epoch_contained(epoch_a, epoch_c)
    assert np.all(actual == expected)

    expected = np.array([
        False,
        True,
        False,
    ])
    actual = epoch_contained(epoch_c, epoch_a)
    assert np.all(actual == expected)


def test_adjust_epoch_bounds(epoch_a):
    expected = epoch_a + np.array([-1, 0])
    actual = adjust_epoch_bounds(epoch_a, -1)
    assert np.all(actual == expected)

    expected = epoch_a + np.array([0, 5])
    actual = adjust_epoch_bounds(epoch_a, 0, 5)
    assert np.all(actual == expected)


def test_remove_overlap():
    epochs = np.array([
        [0, 10],
        [1, 15],
        [5, 10],
        [11, 32],
        [40, 50],
    ])

    expected = np.array([
        [0, 10],
        [11, 32],
        [40, 50],
    ])

    actual = remove_overlap(epochs)
    assert np.all(actual == expected)


def test_find_common_epochs(epoch_df):
    expected = {
        ('parent', 0, 10),
        ('child_a', 0, 1),
        ('child_b', 0, 5),
        ('child_c', 5, 6),
        ('child_d', 6, 10),
    }
    expected = set(expected)

    result = find_common_epochs(epoch_df, 'parent')
    result = set((n, s, e) for n, s, e in result.values)
    assert result == expected


def group_epochs_by_parent(epoch_df):
    result = list(group_epochs_by_parent(epoch_df, r'^PARENT_\d+'))
    n1, df1 = result[0]
    assert n1 == 'parent_1'
    n2, df2 = result[1]
    assert n2 == 'parent_2'


def test_add_epoch(epoch_df):
    result = add_epoch(epoch_df, 'parent', 'child_a')
    assert len(result) == (len(epoch_df) + 2)

    m = result['name'] == 'parent_child_a'
    assert(m.sum() == 2)

    expected = [[1, 2], [30, 31]]
    values = result.loc[m, ['start', 'end']].values
    assert np.array_equal(expected, values)
