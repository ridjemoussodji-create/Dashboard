import pandas as pd
from search_engine import search_engine


def test_numeric_filter():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })

    res = search_engine.advanced_search(df, query='', filters={'value': 20})
    assert len(res) == 1
    assert int(res.iloc[0]['id']) == 2


def test_date_filter_and_fuzzy():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'title': ['Alpha Centauri', 'Betelgeuse', 'Alpha C.']
    })

    # test date filter
    res = search_engine.advanced_search(df, query='', filters={'date': '2020-02-01'})
    assert len(res) == 1
    assert int(res.iloc[0]['id']) == 2

    # test fuzzy search: 'alpha cent' should match rows with Alpha
    res2 = search_engine.advanced_search(df, query='alpha cent', filters={})
    # Expect at least one match containing 'Alpha Centauri' or 'Alpha C.'
    assert len(res2) >= 1
    titles = [str(x).lower() for x in res2['title'].tolist()]
    assert any('alpha' in t for t in titles)
