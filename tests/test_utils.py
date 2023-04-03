from nems0.utils import escaped_split, escaped_join


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

    # after the reverse operation, original string should be preserved
    y = escaped_join(x, '_')
    assert y == s


def test_split_normal_delimiter():
    splits = ['join', 'on', 'me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        s = d.join(splits)
        split = escaped_split(s, d)
        assert split == splits


def test_split_escaped_delimiter():
    splits = ['join', 'on\\', 'me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        s = d.join(splits)
        split = escaped_split(s, d)
        assert split == ['join', f'on\\{d}me', 'test']


def test_join_normal_delimeter():
    splits = ['join', 'on', 'me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        join = escaped_join(splits, d)
        assert join == d.join(splits)


def test_join_escaped_delimeter():
    splits = ['join', 'on\\-', 'me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        join = escaped_join(splits, d)
        assert join == d.join(splits)


def test_join_escaped_delimeter2():
    splits = ['join', 'on', '\\-me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        join = escaped_join(splits, d)
        assert join == d.join(splits)


def test_join_escaped_delimeter3():
    splits = ['join', 'on', '\\--me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        join = escaped_join(splits, d)
        assert join == d.join(splits)


def test_join_escaped_delimeter4():
    splits = ['join', 'on-', '\\-me', 'test']
    delims = ['.', '_', '-']

    for d in delims:
        join = escaped_join(splits, d)
        assert join == d.join(splits)


def test_split_and_join_normal():
    s = 'split_me_on_underscores'
    assert escaped_join(escaped_split(s, '_'), '_') == s


def test_split_and_join_escaped():
    s = r'split_me_on\_underscores'
    assert escaped_join(escaped_split(s, '_'), '_') == s

    s = r'split_me_on_\_underscores'
    assert escaped_join(escaped_split(s, '_'), '_') == s

    s = r'split_me_on__\underscores'
    assert escaped_join(escaped_split(s, '_'), '_') == s
