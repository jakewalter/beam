import os
import importlib
import beam_driver


def test_apply_gpu_safety_factor_sets_env_and_module(monkeypatch):
    # ensure clean environment
    old_env = os.environ.get('BEAM_GPU_SAFETY_FACTOR')
    import beam.core.gpu_beam as gb
    old_mod_val = getattr(gb, 'GPU_MEMORY_SAFETY_FACTOR', None)

    try:
        beam_driver.apply_gpu_safety_factor(5)
        assert os.environ.get('BEAM_GPU_SAFETY_FACTOR') == '5'
        assert getattr(gb, 'GPU_MEMORY_SAFETY_FACTOR') == 5
    finally:
        # restore old env and mod values
        if old_env is None:
            os.environ.pop('BEAM_GPU_SAFETY_FACTOR', None)
        else:
            os.environ['BEAM_GPU_SAFETY_FACTOR'] = old_env
        if old_mod_val is None:
            try:
                delattr(gb, 'GPU_MEMORY_SAFETY_FACTOR')
            except Exception:
                pass
        else:
            gb.GPU_MEMORY_SAFETY_FACTOR = old_mod_val
