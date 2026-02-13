import os
import importlib
import logging
import types
import numpy as np

from beam.core import gpu_beam


class FakeCudaRuntime:
    def __init__(self, free_gb=10, total_gb=16):
        self.free = int(free_gb * 1024**3)
        self.total = int(total_gb * 1024**3)

    def memGetInfo(self):
        return (self.free, self.total)


class FakeCupy:
    def __init__(self, free_gb=10, total_gb=16):
        self.cuda = types.SimpleNamespace(runtime=FakeCudaRuntime(free_gb, total_gb))
        self.fft = np.fft
        self._default_memory_pool = types.SimpleNamespace(free_all_blocks=lambda: None)

    def asarray(self, arr, dtype=None):
        return np.asarray(arr, dtype=dtype)

    def asnumpy(self, arr):
        return arr

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def exp(self, v):
        return np.exp(v)

    def fft_rfft(self, arr, n, axis):
        return np.fft.rfft(arr, n=n, axis=axis)


def test_batching_varies_with_safety_factor(monkeypatch, caplog):
    # Replace real cupy with a fake that reports 10 GB free memory
    fake_cp = FakeCupy(free_gb=10, total_gb=16)
    monkeypatch.setattr(gpu_beam, 'cp', fake_cp)
    monkeypatch.setattr(gpu_beam, 'CUPY_AVAILABLE', True)
    monkeypatch.setattr(gpu_beam, '_GPU_OK', True)

    traces = np.random.randn(4, 2048).astype(np.float32)
    delays = np.zeros((2, 4), dtype=np.float32)

    caplog.set_level(logging.INFO)

    os.environ['BEAM_GPU_SAFETY_FACTOR'] = '4'
    importlib.reload(gpu_beam)
    caplog.clear()
    gpu_beam.beamform_freq_domain(traces, delays, 100.0, use_gpu=True)
    logs4 = [r.message for r in caplog.records if 'After loading data' in r.message]
    assert any('safety_factor=4' in m for m in logs4)

    os.environ['BEAM_GPU_SAFETY_FACTOR'] = '2'
    importlib.reload(gpu_beam)
    caplog.clear()
    gpu_beam.beamform_freq_domain(traces, delays, 100.0, use_gpu=True)
    logs2 = [r.message for r in caplog.records if 'After loading data' in r.message]
    assert any('safety_factor=2' in m for m in logs2)

    # Ensure usable_mem increased when safety factor decreased
    def parse_usable(msg):
        # 'After loading data: 10.000 GB free, safety_factor=2, usable_mem=4.902 GB, ...'
        p = msg.split(',')
        for seg in p:
            if 'usable_mem=' in seg:
                return float(seg.split('=')[1].split()[0])
        return 0.0

    u4 = parse_usable(logs4[0])
    u2 = parse_usable(logs2[0])
    assert u2 > u4
