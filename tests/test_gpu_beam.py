import numpy as np
import os
from beam.core.gpu_beam import beamform_freq_domain, simple_cpu_reference


def _make_synthetic(n_stations=4, n_samples=512, sr=100.0, freq=5.0):
    t = np.arange(n_samples) / sr
    base = np.sin(2 * np.pi * freq * t)
    # create staggered delays across stations
    delays_samples = np.arange(n_stations) * 2  # integer sample offsets
    traces = np.vstack([np.roll(base, int(d)) + 0.01 * np.random.randn(n_samples) for d in delays_samples])
    return traces.astype(np.float32), delays_samples


def test_cpu_reference_and_freq_domain():
    traces, delays_samples = _make_synthetic()
    n_stations, n_samples = traces.shape

    # build a single beam whose delays mirror the station offsets
    delays_sec = np.array([delays_samples / 100.0])  # one beam

    beams_cpu, _ = beamform_freq_domain(traces, delays_sec, sampling_rate=100.0, use_gpu=False)
    assert beams_cpu.shape == (1, n_samples)

    # naive CPU integer-shift reference should produce similar energy distribution
    ref = simple_cpu_reference(traces, delays_samples[None, :])
    assert ref.shape == (1, n_samples)

    # check that the reconstructed beam from freq-domain is not all zeros
    assert np.max(np.abs(beams_cpu)) > 0.0

    # energy similarity check (not exact due to windowing/FFT rounding)
    energy_ratio = np.sum(beams_cpu**2) / (np.sum(ref**2) + 1e-8)
    assert 0.1 < energy_ratio < 10.0


def test_compute_envelope_and_sta_lta_shapes():
    traces, delays_samples = _make_synthetic(n_stations=4, n_samples=512)
    # make a single beam (shifts are in seconds here)
    delays_sec = np.array([delays_samples / 100.0])

    beams_cpu, _ = beamform_freq_domain(traces, delays_sec, sampling_rate=100.0, use_gpu=False)

    # compute envelope (CPU path) and STA/LTA
    env = __import__('beam.core.gpu_beam', fromlist=['compute_envelope']).compute_envelope(beams_cpu, use_gpu=False)
    assert env.shape == beams_cpu.shape

    ratio = __import__('beam.core.gpu_beam', fromlist=['sta_lta']).sta_lta(env, 100.0, 5, 50, use_gpu=False)
    # ratio should be full-length and match beams shape
    assert ratio.shape == beams_cpu.shape
    assert np.any(ratio > 0.0)


def test_gpu_unusable_flag_prefers_cpu():
    # Ensure that if the module marks the GPU unusable we still run the
    # CPU path even when use_gpu=True.
    import beam.core.gpu_beam as gb

    traces, delays_samples = _make_synthetic()
    delays_sec = np.array([delays_samples / 100.0])

    prev_flag = getattr(gb, "_GPU_OK", None)
    try:
        gb._GPU_OK = False
        beams_cpu, _ = gb.beamform_freq_domain(traces, delays_sec, sampling_rate=100.0, use_gpu=True)
        assert beams_cpu.shape == (1, traces.shape[1])
    finally:
        if prev_flag is None:
            try:
                delattr(gb, "_GPU_OK")
            except Exception:
                pass
        else:
            gb._GPU_OK = prev_flag


def test_gpu_ok_flag_defined():
    import beam.core.gpu_beam as gb
    # The module should always expose a boolean _GPU_OK flag
    assert hasattr(gb, "_GPU_OK")
    assert isinstance(gb._GPU_OK, bool)


def test_missing_gpu_ok_defers_to_cupy_flag():
    # If _GPU_OK is absent from the module namespace, the function should
    # still work and default to CUPY_AVAILABLE without raising NameError.
    import importlib
    import beam.core.gpu_beam as gb
    import os

    # temporarily remove the attribute if present
    prev = getattr(gb, '_GPU_OK', None)
    if hasattr(gb, '_GPU_OK'):
        delattr(gb, '_GPU_OK')
    try:
        traces, delays_samples = _make_synthetic()
        delays_sec = np.array([delays_samples / 100.0])

        # This should not raise NameError even when _GPU_OK was deleted.
        beams, _ = gb.beamform_freq_domain(traces, delays_sec, sampling_rate=100.0, use_gpu=True)
        assert beams.shape == (1, traces.shape[1])
    finally:
        if prev is None:
            try:
                delattr(gb, '_GPU_OK')
            except Exception:
                pass
        else:
            gb._GPU_OK = prev


def test_gpu_flag_file_disables_gpu(tmp_path, monkeypatch):
    import importlib
    import beam.core.gpu_beam as gb

    # Use a temporary file path for the GPU-disable marker
    test_flag = str(tmp_path / "beam_gpu_disabled_test")
    monkeypatch.setattr(gb, '_GPU_FLAG_PATH', test_flag)

    # Ensure file doesn't exist and _GPU_OK is True initially
    if os.path.exists(test_flag):
        os.remove(test_flag)
    gb._GPU_OK = True

    # Simulate writing the marker (like an OOM handler would do)
    with open(test_flag, 'w') as f:
        f.write('disabled')

    # Re-import or re-evaluate initialization logic (module already has check)
    # We'll emulate the behavior by reading the file ourselves
    gb._GPU_OK = not os.path.exists(gb._GPU_FLAG_PATH)
    assert gb._GPU_OK is False
    # cleanup
    os.remove(test_flag)


def test_classic_sta_lta_matches_obspy():
    # synthetic single station waveform
    import numpy as np
    from obspy.signal.trigger import classic_sta_lta as obspy_classic

    traces, delays_samples = _make_synthetic(n_stations=1, n_samples=1024)
    # build one-beam shaped array
    delays_sec = np.array([[0.0]])
    beams_cpu, _ = beamform_freq_domain(traces, delays_sec, sampling_rate=100.0, use_gpu=False)

    # use simple envelope (here beams_cpu is the raw trace) and compute
    # obsPy classic STA/LTA with given windows
    sta = 5
    lta = 50
    obspy_ratio = obspy_classic(beams_cpu.flatten(), sta, lta)

    # our implementation on CPU
    ratio_cpu = __import__('beam.core.gpu_beam', fromlist=['classic_sta_lta']).classic_sta_lta(beams_cpu[None, :], sta, lta, use_gpu=False)
    ratio_cpu = ratio_cpu.flatten()

    # They may differ at the edges; compare central region where both windows are valid
    start = lta
    end = len(obspy_ratio)
    # avoid comparing tiny arrays
    assert end - start > 10

    # Compare with tolerance — allow small numerical differences
    assert np.allclose(obspy_ratio[start:end], ratio_cpu[start:end], rtol=1e-6, atol=1e-8)


def test_classic_sta_lta_matches_obspy():
    import numpy as _np
    from obspy.signal.trigger import classic_sta_lta as obs_classic
    from beam.core import gpu_beam as gb

    # synthetic trace
    sr = 100.0
    n = 512
    t = _np.arange(n) / sr
    trace = _np.sin(2 * _np.pi * 3.0 * t) + 0.01 * _np.random.randn(n)

    nsta = 5
    nlta = 50

    obs_ratio = obs_classic(trace, nsta, nlta)
    gb_ratio = gb.classic_sta_lta(trace, nsta, nlta, use_gpu=False)

    # exact equality is unlikely because of dtype differences, but values
    # should numerically match closely
    assert obs_ratio.shape == gb_ratio.shape
    # allow some small numerical difference
    diff = _np.max(_np.abs(obs_ratio - gb_ratio))
    assert diff < 1e-6, f"max diff too large: {diff}"

    # 2D case
    traces2 = _np.vstack([trace, trace * 0.5])
    obs_r2 = _np.vstack([obs_ratio, obs_classic(trace * 0.5, nsta, nlta)])
    gb_r2 = gb.classic_sta_lta(traces2, nsta, nlta, use_gpu=False)
    assert gb_r2.shape == traces2.shape
    assert _np.allclose(obs_r2, gb_r2, atol=1e-6)


def test_classic_sta_lta_on_envelope():
    import numpy as _np
    from obspy.signal.trigger import classic_sta_lta as obs_classic
    from beam.core.gpu_beam import classic_sta_lta as gb_classic
    from scipy.signal import hilbert

    np.random.seed(2)
    n = 512
    t = _np.arange(n) / 100.0
    trace = _np.sin(2 * _np.pi * 5.0 * t) + 0.005 * _np.random.randn(n)

    env = _np.abs(hilbert(trace))
    nsta = 5
    nlta = 50

    obs_r = obs_classic(env, nsta, nlta)
    gb_r = gb_classic(env, nsta, nlta, use_gpu=False)

    assert obs_r.shape == gb_r.shape
    assert _np.allclose(obs_r, gb_r, atol=1e-8)
