import numpy as np
from beam.core import gpu_beam as gb


def test_beamform_returns_tuple_cpu_and_gpu():
    # Small synthetic dataset
    n_stations = 4
    n_samples = 2048
    n_beams = 8

    traces = np.random.randn(n_stations, n_samples).astype(np.float32)
    # delays for each beam (seconds) per station
    delays = np.zeros((n_beams, n_stations), dtype=np.float32)

    # Introduce small delays to make beams different
    for b in range(n_beams):
        delays[b, :] = (b * 0.02) * np.arange(n_stations)

    # CPU path
    beams_cpu, freqs_cpu = gb.beamform_freq_domain(traces, delays, sampling_rate=100.0, use_gpu=False)
    assert beams_cpu is not None, "CPU beamformer returned None"
    assert isinstance(beams_cpu, np.ndarray)
    assert beams_cpu.shape == (n_beams, n_samples)
    assert freqs_cpu is not None

    # GPU path (if available) - permit fallback to CPU but ensure we never get None
    try:
        beams_gpu, freqs_gpu = gb.beamform_freq_domain(traces, delays, sampling_rate=100.0, use_gpu=True)
        assert beams_gpu is not None, "GPU beamformer returned None"
        # beams_gpu might be device array or host array; ensure we can read shape
        assert hasattr(beams_gpu, 'shape')
        assert freqs_gpu is not None
    except RuntimeError:
        # If GPU cannot be used and the function raises a RuntimeError for device array requirement,
        # it's okay; otherwise GPU fallback should not return None
        pass
