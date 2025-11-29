import numpy as np

from dRFEtools.lowess import redundant


def sample_elimination_dict():
    return {
        10: {"metrics": {"nmi_score": 0.6, "accuracy_score": 0.7}},
        5: {"metrics": {"nmi_score": 0.65, "accuracy_score": 0.72}},
        2: {"metrics": {"nmi_score": 0.5, "accuracy_score": 0.68}},
    }


def test_elimination_dataframe_sorted():
    df = redundant._get_elim_df_ordered(
        sample_elimination_dict(), multi=False, use_accuracy=False
    )
    assert df.iloc[0]["x"] == 2
    assert np.all(np.diff(df["x"].to_numpy()) > 0)


def test_lowess_outputs_shapes():
    d = sample_elimination_dict()
    x, y, z, xnew, ynew = redundant._cal_lowess(d, redundant.DEFAULT_FRAC,
                                                False, False)
    assert len(x) == len(y)
    assert len(xnew) == redundant.LOWESS_POINTS
    assert isinstance(z, np.ndarray)
