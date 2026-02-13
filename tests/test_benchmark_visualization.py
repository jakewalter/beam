import os

from beam.visualization import plot_benchmark_comparison


def test_plot_benchmark_comparison(tmp_path):
    outdir = str(tmp_path)

    results = [
        {'label': 'cpu', 'wall_time': 120.0, 'unique_detections': 100, 'per_subarray': {0: 50, 1: 50}},
        {'label': 'gpu', 'wall_time': 80.0, 'unique_detections': 110, 'per_subarray': {0: 60, 1: 50}}
    ]

    files = plot_benchmark_comparison(results, outdir=outdir, prefix='testbench')
    assert 'summary' in files and os.path.exists(files['summary'])
    assert 'scatter' in files and os.path.exists(files['scatter'])
    assert 'per_subarray' in files and os.path.exists(files['per_subarray'])
