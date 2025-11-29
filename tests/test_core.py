from dRFEtools.dRFEtools import _n_features_iter


def test_n_features_iter_respects_keep_rate():
    sequence = list(_n_features_iter(10, 0.5))
    assert sequence[-1] == 1
    assert sequence[0] == 5
    assert all(n > 0 for n in sequence)
