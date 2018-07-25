from nems.utils import escaped_split, escaped_join

def test_split_and_join():
    s = 'my-escaped_modelspec-name_blah.t5\.5'
    x = escaped_split(s, '_')
    # escaping the . shouldn't matter here, should only escape
    # the active delimiter
    assert len(x) == 3
    blah = x[2]

    # now the . should be ignored during the split, for a length
    # of 2 instead of 3
    assert len(escaped_split(blah, '.')) == 2

    # after the reverse operation, original string shouldl be preserved
    y = escaped_join(x, '_')
    assert y == s
