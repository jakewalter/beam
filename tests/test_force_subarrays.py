from beam_driver import TraditionalBeamformer


def test_force_subarrays_even_split():
    bf = TraditionalBeamformer(data_dir='.')
    coords = {f'ST{i:02d}': (0.0 + 0.01*i, -120.0 + 0.01*i, 0.0) for i in range(8)}
    bf.set_station_coords(coords)
    groups = bf.force_subarrays(2)
    assert len(groups) == 2
    # ensure the group sizes are balanced (4 and 4)
    assert sorted([len(g) for g in groups]) == [4, 4]


def test_force_subarrays_more_groups_than_stations():
    bf = TraditionalBeamformer(data_dir='.')
    coords = {'A': (0,0,0), 'B': (0.01,0,0)}
    bf.set_station_coords(coords)
    groups = bf.force_subarrays(4)
    # should create as many groups as stations (2)
    assert len(groups) == 2