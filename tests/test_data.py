from src.data import make_dataset, FEATURES

def test_dataset_is_reproducible_and_well_formed():
    a = make_dataset(n=100, random_state=7)
    b = make_dataset(n=100, random_state=7)
    assert a.equals(b)
    assert set(FEATURES).issubset(a.columns)
    assert set(a["bad_90dpd"].unique()).issubset({0, 1})
