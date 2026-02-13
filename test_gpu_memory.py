#!/usr/bin/env python3
"""
Quick test to verify GPU memory calculations work correctly.
Run this to check if the batch sizing is reasonable for your GPU.
"""
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - GPU testing not possible")
    exit(0)

def test_memory_calculation():
    """Simulate the memory calculation logic"""
    # Get current GPU memory
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    print(f"GPU Memory: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
    
    # Simulate a realistic seismic array scenario
    n_stations = 20
    n_samples = 600000  # 100 Hz * 6000 seconds = ~1.67 hours
    n_beams = 1296  # 360 azimuths * 5 degree spacing * some slowness values
    
    # Calculate FFT size
    nfft = 2 ** int(np.ceil(np.log2(n_samples)))
    nf = nfft // 2 + 1
    
    print(f"\nScenario: {n_stations} stations, {n_samples} samples, {n_beams} beams")
    print(f"FFT size: {nfft}, positive frequencies: {nf}")
    
    # Memory estimates (same as in gpu_beam.py)
    SAFETY = 0.25
    usable_mem = int(free_mem * SAFETY)
    print(f"Usable memory (with {SAFETY*100:.0f}% safety factor): {usable_mem / (1024**3):.3f} GB")
    
    bytes_traces = n_stations * n_samples * 4  # float32
    bytes_X = n_stations * nf * 8  # complex64
    per_beam_bytes = n_stations * 4 + nf * 16 + nfft * 4
    headroom = int(50 * 1024 * 1024)  # 50MB
    
    print(f"\nMemory requirements:")
    print(f"  Traces:  {bytes_traces / (1024**2):.1f} MB")
    print(f"  FFT(X):  {bytes_X / (1024**2):.1f} MB")
    print(f"  Per beam: {per_beam_bytes / (1024**2):.3f} MB")
    print(f"  Headroom: {headroom / (1024**2):.1f} MB")
    
    available_for_batching = usable_mem - bytes_traces - bytes_X - headroom
    
    if available_for_batching <= 0:
        print(f"\nINSUFFICIENT MEMORY: Need {(bytes_traces + bytes_X + headroom) / (1024**3):.3f} GB minimum")
        print("GPU processing not possible with current memory constraints")
        return
    
    batch_size = max(1, int(available_for_batching / per_beam_bytes))
    batch_size = min(batch_size, n_beams)
    
    print(f"\nCalculated batch_size: {batch_size} beams per batch")
    print(f"Number of batches needed: {int(np.ceil(n_beams / batch_size))}")
    print(f"Memory per batch: {(batch_size * per_beam_bytes) / (1024**2):.1f} MB")
    
    # Try to actually allocate to verify
    print("\nTrying actual allocation...")
    try:
        cp._default_memory_pool.free_all_blocks()
        
        # Allocate traces and FFT
        d_tr = cp.random.randn(n_stations, n_samples, dtype=cp.float32)
        X = cp.fft.rfft(d_tr, n=nfft, axis=1)
        
        # Try one batch
        summed = cp.zeros((batch_size, nf), dtype=cp.complex64)
        d_batch = cp.random.randn(batch_size, n_stations, dtype=cp.float32)
        phase_s = cp.random.randn(batch_size, nf, dtype=cp.complex64)
        beams_freq = cp.random.randn(batch_size, nfft, dtype=cp.float32)
        
        print("✓ Allocation successful!")
        print(f"Actual memory used: {(cp.get_default_memory_pool().used_bytes()) / (1024**3):.3f} GB")
        
        # Clean up
        del d_tr, X, summed, d_batch, phase_s, beams_freq
        cp._default_memory_pool.free_all_blocks()
        
    except Exception as e:
        print(f"✗ Allocation failed: {e}")

if __name__ == "__main__":
    test_memory_calculation()
