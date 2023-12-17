def test_closest_multiple():
    from common import closest_multiple
    assert closest_multiple(3, 100) == 99
    assert closest_multiple(4, 100) == 100
    assert closest_multiple(30, 100) == 90
