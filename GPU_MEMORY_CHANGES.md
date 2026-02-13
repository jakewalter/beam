# GPU Memory Optimization Changes

## Summary
Refactored GPU beamforming to use memory more efficiently and avoid OOM errors by checking available memory BEFORE allocating data, rather than after. Also optimized CPU fallback to handle large beam counts without exhausting system RAM.

## Key Changes

### 1. Memory Check Before Allocation
- **Old approach**: Load data → FFT → check memory → try to batch → OOM
- **New approach**: Check memory → calculate safe batch size → load data → batch without issues

### 2. Conservative Safety Factor
- Use only 25% of reported free GPU memory (equivalently require at least 4× the estimated persistent data size)
- Accounts for:
  - Memory fragmentation
  - Other processes using GPU
  - CuPy memory pool overhead
  - Temporary allocations during computation

### 3. Better Memory Estimation
More accurate per-beam memory calculation:
```python
per_beam_bytes = n_stations * 4 + nf * 16 + nfft * 4
# Accounts for:
# - d_batch (delays): batch_size * n_stations * 4 bytes
# - summed: batch_size * nf * 8 bytes (complex64)
# - phase_s: batch_size * nf * 8 bytes (complex64)
# - beams_freq: batch_size * nfft * 4 bytes (float32)
```

### 4. Removed Retry/Fallback Logic
- **Old**: Try GPU → OOM → retry with smaller batch → OOM → retry → eventually give up
- **New**: Calculate correct batch size upfront, execute once
- Falls back to CPU only if memory check shows GPU can't handle even the persistent data
- Raises clear error if OOM happens despite conservative estimates

### 5. Optimized CPU Fallback
- **Old**: Pre-allocate full (n_beams × n_samples) array → OOM for large datasets
- **New**: Process beams in chunks of 64 → vstack chunks at the end
- For 720 beams × 8.64M samples: 
  - Old: 24.9 GB allocation (fails)
  - New: 12 chunks of 2.2 GB each (succeeds)

### 6. Better Logging
- Logs calculated batch size and available memory
- Clear warning if falling back to CPU due to insufficient memory
- Informative error messages if GPU fails

## Testing

All 21 tests pass, including:
- `test_gpu_beam.py`: GPU-specific beamforming tests
- `test_integration_gpu_triangulation.py`: End-to-end GPU workflow
- All other existing tests

Run `python3 test_gpu_memory.py` to check if your GPU has sufficient memory for a realistic scenario.

## Expected Behavior

### Sufficient GPU Memory
```
GPU memory: 6.50 GB free, calculated batch_size=128 for 1296 beams
```
Processing will use GPU with the calculated batch size.

### Insufficient GPU Memory
```
WARNING: Insufficient GPU memory: 0.13 GB free, need ~0.172 GB just for data. Using CPU instead.
```
Processing will automatically fall back to CPU without errors.

### OOM Despite Conservative Estimates
```
ERROR: GPU Out of Memory with batch_size=X for Y beams
RuntimeError: GPU ran out of memory (batch_size=X, n_beams=Y). The GPU may not have enough memory for this operation.
```
This should rarely happen with the 25% safety factor (4×), but if it does, it indicates:
- Other processes grabbed GPU memory during processing
- Memory fragmentation issues
- Extremely large data size

## User Command
The full June-August 2020 processing command should now work:
```bash
PYTHONPATH=. python3 beam_driver.py --mode traditional \
  -d /scratch2/time/day_volumes \
  -s 20200601 -e 20200831 \
  --force-subarrays 2 \
  --az-step 5.0 \
  --gpu \
  --fk-max-per-subarray 3 \
  --fk-min-snr 10 \
  --min-snr-output 10 \
  --plot \
  --plot-dir /scratch2/time/day_volumes/bench_june2020_snr10 \
  -p 1 \
  -o /scratch2/time/day_volumes/bench_june2020_snr10/detections.txt \
  --inventory /scratch2/time/day_volumes/time.xml
```
