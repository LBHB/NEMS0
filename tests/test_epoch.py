import pytest

import numpy as np
import pandas as pd

from nems.epoch import (epoch_union, epoch_difference, epoch_intersection,
                        epoch_contains, epoch_contained, adjust_epoch_bounds,
                        remove_overlap, find_common_epochs)

@pytest.fixture()
def epoch_a():
    return np.array([
        [  0,  50],
        [ 60,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
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


def test_intersection(epoch_a, epoch_b):
    expected = np.array([
        [ 60,  70],
        [ 75,  76],
        [ 90,  95],
    ])
    result = epoch_intersection(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_empty_intersection():
    a = np.array([
        [  0, 20],
        [ 50, 70]
    ])

    b = np.array([
        [ 25, 30],
        [ 30, 45]
    ])

    with pytest.raises(RuntimeWarning,
                       message="Expected RuntimeWarning for size 0"):
        result = epoch_intersection(a, b)


def test_union(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 55,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
        [110, 120],
    ])
    result = epoch_union(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_difference(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 77,  77],
        [ 85,  90],
        [ 95, 100],
    ])
    result = epoch_difference(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_empty_difference():
    a = b = np.array([
        [  0,  50],
        [ 50, 100]
    ])

    with pytest.raises(RuntimeWarning,
                       message="Expected RuntimeWarning for size 0"):
        result = epoch_difference(a, b)


def test_contains(epoch_a, epoch_b):
    expected_any = np.array([
        False,
        True,
        True,
        False,
        True,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'any')
    assert np.all(actual == expected_any)

    expected_start = np.array([
        False,
        False,
        True,
        False,
        True,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'start')
    assert np.all(actual == expected_start)

    expected_end = np.array([
        False,
        True,
        True,
        False,
        True,
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


def test_find_common_epochs():
    epochs = [
        ['parent', 1, 11],
        ['child_a', 1, 2],
        ['child_b', 1, 6],
        ['child_c', 6, 7],
        ['child_d', 7, 11],
        ['child_e', 9, 11],
        ['parent', 30, 40],
        ['child_a', 30, 31],
        ['child_b', 30, 35],
        ['child_c', 35, 36],
        ['child_d', 36, 40],
    ]
    epochs = pd.DataFrame(epochs, columns=['name', 'start', 'end'])

    expected = {
        ('parent', 0, 10),
        ('child_a', 0, 1),
        ('child_b', 0, 5),
        ('child_c', 5, 6),
        ('child_d', 6, 10),
    }
    expected = set(expected)

    result = find_common_epochs(epochs, 'parent')
    result = set((n, s, e) for n, s, e in result.values)
    assert result == expected
