import os
from beam_driver import TraditionalBeamformer, BeamArrayDetector

def test_save_detections_traditional(tmp_path):
    detections = {
        '20200101': [
            {'time': '2020-01-01T00:00:00Z', 'velocity': 3.0, 'backazimuth': 0.0, 'snr': 5.0, 'duration': 1.0},
            {'time': '2020-01-01T00:01:00Z', 'velocity': 4.0, 'backazimuth': 90.0, 'snr': 12.0, 'duration': 2.0},
        ]
    }

    bf = TraditionalBeamformer(data_dir='.')
    out = tmp_path / 'out.txt'
    bf.save_detections(detections, str(out), min_snr=10.0)

    txt = out.read_text()
    assert '00:00:00' not in txt
    assert '00:01:00' in txt


def test_save_detections_correlation(tmp_path):
    detections = {
        '20200101': [
            {'time': '2020-01-01T00:00:00Z', 'array_correlation': 0.5, 'scaled_correlation': 0.2, 'n_channels': 5, 'snr': 3.0},
            {'time': '2020-01-01T00:02:00Z', 'array_correlation': 0.9, 'scaled_correlation': 0.8, 'n_channels': 7, 'snr': 15.0},
        ]
    }

    det = BeamArrayDetector(data_dir='.')
    out = tmp_path / 'out_corr.txt'
    det.save_detections(detections, str(out), min_snr=10.0)

    txt = out.read_text()
    assert '00:00:00' not in txt
    assert '00:02:00' in txt
