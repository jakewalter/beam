"""GPU-backed beamforming utilities (CuPy prototype)

This module provides minimal helpers to run delay-and-sum beamforming
on a GPU using CuPy. The implementation is intentionally small and
straightforward so it can be used as a drop-in prototype and compared
to the existing CPU code.

Design notes
- We implement a frequency-domain phase-shift beamformer that supports
  fractional-sample delays. It accepts the array of traces and a
  delays matrix (n_beams x n_stations in seconds) and produces beams
  for each beam in parallel on the GPU.
- The functions attempt to import CuPy; if unavailable they fall back
  to NumPy-based computation so tests can still run on CPU-only systems.
"""

from typing import Tuple
import logging
import numpy as np
import os
import tempfile

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# Track whether the GPU has previously failed during this process. If the
# GPU has already OOM'd then subsequent calls will prefer CPU to avoid
# repeated failures and noisy logs. Always create the flag so it's defined
# regardless of whether CuPy was importable.
_GPU_OK = CUPY_AVAILABLE

# Cross-process marker path. When a process experiences a hard GPU OOM and
# decides the device is unusable, we write a small marker file here so other
# processes in the same machine/user prefer CPU instead of repeatedly hitting
# the same OOM condition.
_GPU_FLAG_PATH = os.path.join(tempfile.gettempdir(), f"beam_gpu_disabled_{os.getuid()}")

# Initialize _GPU_OK based on the presence of the marker file.
if os.path.exists(_GPU_FLAG_PATH):
    _GPU_OK = False

# Default safety factor multiplier used when computing estimated persistent
# GPU memory required for the operation. This is conservative by default
# and protects against temporary allocations, CuPy pool overhead and
# fragmentation. It can be overridden by env var BEAM_GPU_SAFETY_FACTOR.
GPU_MEMORY_SAFETY_FACTOR = int(os.getenv('BEAM_GPU_SAFETY_FACTOR', '4'))


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def beamform_freq_domain(traces: np.ndarray,
                         delays_sec: np.ndarray,
                         sampling_rate: float,
                         use_gpu: bool = True,
                         return_device_array: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency-domain, batched delay-and-sum beamformer.

    Parameters
    ----------
    traces : np.ndarray
        Array of shape (n_stations, n_samples), dtype float32 or float64.
    delays_sec : np.ndarray
        Array of shape (n_beams, n_stations) giving delay for each station
        (in seconds) for every beam.
    sampling_rate : float
        Sampling rate (Hz) of the traces.
    use_gpu : bool
        Attempt to run on GPU if CuPy is available. Falls back to CPU.

    Returns
    -------
    beams : np.ndarray
        Array of shape (n_beams, n_samples) with reconstructed beam traces
        (real-valued). Returned as NumPy arrays even when computed on GPU.
    freqs : np.ndarray
        Frequency vector corresponding to the positive-frequency FFT bins
        used internally (useful for debugging).
    """
    
    logger = logging.getLogger(__name__)

    # Validate input shapes
    traces = np.asarray(traces)
    delays_sec = np.asarray(delays_sec)

    if traces.ndim != 2:
        raise ValueError("traces must be 2D (n_stations, n_samples)")

    n_stations, n_samples = traces.shape
    n_beams, ds = delays_sec.shape
    if ds != n_stations:
        raise ValueError("delays_sec must be shape (n_beams, n_stations)")

    logger.debug(f"beamform_freq_domain called with {n_stations} stations, {n_samples} samples, {n_beams} beams, use_gpu={use_gpu}")
    logger.debug("ENTER beamform_freq_domain: n_stations=%d n_samples=%d n_beams=%d use_gpu=%s return_device_array=%s",
                 n_stations, n_samples, n_beams, use_gpu, return_device_array)

    n_stations, n_samples = traces.shape
    n_beams, ds = delays_sec.shape
    if ds != n_stations:
        raise ValueError("delays_sec must be shape (n_beams, n_stations)")

    # Choose FFT length (use nfft >= n_samples; next pow2 helps performance)
    nfft = _next_pow2(n_samples)

    # rfft length
    nf = nfft // 2 + 1

    # choose device arrays
    global _GPU_OK
    # Defensive guard: if for some reason _GPU_OK hasn't been set at
    # module-import time (e.g., weird import fiddling), initialize it to
    # the current CUPY_AVAILABLE value so the expression below doesn't
    # raise NameError.
    if '_GPU_OK' not in globals():
        _GPU_OK = CUPY_AVAILABLE

    # Check GPU memory availability BEFORE loading any data to device.
    # User requirement: "batch it to fit the gpu free memory" - use conservative estimates.
    logger = logging.getLogger(__name__)
    batch_size = None  # Will be calculated based on available memory
    
    if use_gpu and CUPY_AVAILABLE and _GPU_OK:
        try:
            free_mem_initial, total_mem = cp.cuda.runtime.memGetInfo()
            
            # Estimate bytes for persistent data (traces + FFT)
            # These are conservative estimates - actual usage may be MUCH higher due to:
            # - Temporary arrays during FFT (CuPy can use 2-4x)
            # - CuPy memory pool caching
            # - Memory fragmentation
            bytes_traces = n_stations * n_samples * 4  # float32
            bytes_X = n_stations * nf * 8  # complex64
            
            # Read safety factor dynamically so runtime changes to the
            # environment or driver-level overrides are applied even if
            # the module was imported earlier.
            try:
                current_safety_factor = int(os.getenv('BEAM_GPU_SAFETY_FACTOR', str(GPU_MEMORY_SAFETY_FACTOR)))
            except Exception:
                current_safety_factor = GPU_MEMORY_SAFETY_FACTOR
            # log current factor being used for debugging/visibility
            logger.debug("Using GPU_MEMORY_SAFETY_FACTOR=%d (module default %d)", current_safety_factor, GPU_MEMORY_SAFETY_FACTOR)

            # Require a conservative safety factor (configurable); default changed to 4×
            min_required = (bytes_traces + bytes_X) * current_safety_factor
            
            if free_mem_initial < min_required:
                logger.warning(
                    "Insufficient GPU memory: %.2f GB free, need ~%.3f GB (%d× data size for safety). "
                    "Using CPU instead.",
                    free_mem_initial / (1024**3),
                    min_required / (1024**3),
                    current_safety_factor
                )
                use_gpu = False
            else:
                # We'll calculate batch_size after loading data to see actual free memory
                logger.debug(
                    "GPU has %.2f GB free (need %.2f GB), proceeding to load data",
                    free_mem_initial / (1024**3),
                    min_required / (1024**3)
                )
            
        except RuntimeError:
            # Re-raise memory errors - don't fall back to CPU
            raise
        except Exception as e:
            # For other GPU errors, warn and fall back
            logger.warning("GPU initialization failed: %s, using CPU", e)
            use_gpu = False

    # If the caller asked for a device-array result but we won't be using
    # the GPU, refuse early so the caller can explicitly retry on CPU.
    if return_device_array and not (use_gpu and CUPY_AVAILABLE and _GPU_OK):
        raise RuntimeError("caller requested device-array but GPU not available/selected")

    xp = cp if (use_gpu and CUPY_AVAILABLE and _GPU_OK) else np

    # Move traces to device
    if xp is cp:
        d_tr = cp.asarray(traces, dtype=cp.float32)
    else:
        d_tr = traces.astype(np.float32)

    # Compute positive-frequency FFT across samples for all stations
    if xp is cp:
        X = cp.fft.rfft(d_tr, n=nfft, axis=1)
    else:
        X = np.fft.rfft(d_tr, n=nfft, axis=1)

    # Frequency vector
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate).astype(np.float32)
    if xp is cp:
        d_freqs = cp.asarray(freqs)
    else:
        d_freqs = freqs

    # Build phase factors and produce beams. For large numbers of beams
    # the (n_beams, n_stations, nf) allocation can be huge and cause OOM.
    # For GPU execution we therefore chunk the beam dimension into batches
    # and compute a per-batch summation over stations without creating the
    # full 3D phase array. This keeps peak memory low and gracefully
    # handles large beam counts.

    delays = delays_sec.astype(np.float32)

    if xp is cp:
        # NOW check actual free memory after loading data
        # Calculate batch_size based on what's actually available
        try:
            free_mem_after, _ = cp.cuda.runtime.memGetInfo()
            
            # Per-batch working memory (for batch_size beams):
            # - d_batch (delays): batch_size * n_stations * 4 bytes
            # - summed: batch_size * nf * 8 bytes (complex64)
            # - phase_s: batch_size * nf * 8 bytes (complex64) 
            # - beams_freq: batch_size * nfft * 4 bytes (float32)
            per_beam_bytes = n_stations * 4 + nf * 16 + nfft * 4
            
            # Determine usable memory for the per-beam batching calculation.
            # The safe memory usable is inversely proportional to the configured
            # GPU safety factor: a larger safety factor reduces how much free
            # memory we are willing to use for batching to reduce OOM risk.
            safety_factor = max(1, int(current_safety_factor))
            usable_mem = int(free_mem_after / safety_factor)
            headroom = int(100 * 1024 * 1024)  # 100MB headroom
            
            available_for_batching = max(0, usable_mem - headroom)
            
            if available_for_batching < per_beam_bytes:
                msg = (
                    f"After loading data: only {free_mem_after / (1024**3):.3f} GB free, "
                    f"insufficient for even 1 beam (needs {per_beam_bytes / (1024**3):.3f} GB)."
                )
                logger.warning(msg + " Using CPU if possible.")
                # If the caller explicitly requested a device-array we must
                # raise an error so higher-level code can handle it. If not,
                # fall back to CPU by converting arrays to NumPy and switching
                # execution path.
                if return_device_array:
                    raise RuntimeError(f"Insufficient GPU memory after loading data: {free_mem_after / (1024**3):.3f} GB free")
                # Convert device arrays to host arrays and switch to CPU path
                try:
                    X = cp.asnumpy(X)
                except Exception:
                    # If conversion fails, propagate the error upwards
                    raise
                try:
                    d_freqs = cp.asnumpy(d_freqs)
                except Exception:
                    d_freqs = d_freqs if not isinstance(d_freqs, cp.ndarray) else cp.asnumpy(d_freqs)
                xp = np
                # ensure subsequent CPU path runs; do not continue GPU processing
                # fallthrough to CPU branch
            
            batch_size = max(1, int(available_for_batching / per_beam_bytes))
            batch_size = min(batch_size, n_beams)  # Don't exceed total beams
            
            logger.info(
                "After loading data: %.3f GB free, safety_factor=%d, usable_mem=%.3f GB, batch_size=%d for %d beams",
                free_mem_after / (1024**3), safety_factor, available_for_batching / (1024**3), batch_size, n_beams
            )
            
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Could not check GPU memory after loading data: %s", e)
            batch_size = min(1, n_beams)  # Very conservative fallback

        # Only proceed with GPU processing if we are still on the device
        if xp is cp:
            # Prepare device arrays reused across batches
            d_X = X  # already a device array

            # free any cached blocks to give us a cleaner starting point
            try:
                cp._default_memory_pool.free_all_blocks()
            except Exception:
                pass

            # Process beams in batches using calculated batch_size
            beams_dev_list = []
            try:
                    for start in range(0, n_beams, batch_size):
                        end = min(start + batch_size, n_beams)
                        B = end - start
                        # accumulate summed spectrum for batch: shape (B, nf)
                        summed = cp.zeros((B, nf), dtype=cp.complex64)

                        # delays for this batch: shape (B, n_stations)
                        d_batch = cp.asarray(delays[start:end, :])

                        # iterate stations to avoid allocating full (B, n_stations, nf)
                        for s in range(n_stations):
                            # delays for this station across beams -> (B,)
                            dvec = d_batch[:, s]
                            # phase for this station across beams: (B, nf)
                            # compute via outer product of dvec and freqs
                            phase_s = cp.exp(-2j * cp.pi * (d_freqs[None, :] * dvec[:, None]))
                            # multiply with X[s,:] broadcast to (B, nf)
                            summed += phase_s * d_X[s][None, :]

                        # inverse FFT -> time domain beams for batch
                        beams_freq = cp.fft.irfft(summed, n=nfft, axis=1)
                        batch_beams = beams_freq[:, :n_samples]

                        if return_device_array:
                            beams_dev_list.append(batch_beams)
                        else:
                            beams_dev_list.append(batch_beams.get())

                    # stack results and return
                    if return_device_array:
                        beams_dev = cp.concatenate(beams_dev_list, axis=0)
                        return beams_dev, freqs
                    else:
                        beams = np.vstack(beams_dev_list)
                        return beams, freqs

            except cp.cuda.memory.OutOfMemoryError as e:
                    # GPU OOM despite careful planning - this shouldn't happen with our conservative estimates
                    logger.error(
                        "GPU Out of Memory with batch_size=%d for %d beams: %s. "
                        "Try reducing data size or use CPU (remove --gpu flag).",
                        batch_size, n_beams, e
                    )
                    raise RuntimeError(
                        f"GPU ran out of memory (batch_size={batch_size}, n_beams={n_beams}). "
                        f"The GPU may not have enough memory for this operation."
                    ) from e
            except Exception as e:
                    # Other GPU errors
                    logger.error("GPU beamforming failed: %s", e)
                    raise

    # CPU path (only executed if GPU was never selected)
    if xp is np:
        # If the caller asked explicitly for a device-array but we ended up
        # on the CPU (insufficient GPU), we must not silently return host
        # arrays — the caller expects device arrays and will try to run
        # device-only processing. Signal the caller so it can perform an
        # explicit CPU fallback path.
        if return_device_array:
            raise RuntimeError("Requested device-array return but GPU unavailable or insufficient memory")
        logger.info(f"CPU beamforming: processing {n_beams} beams")
        
        # Process beams one at a time, collect results
        beam_list = []
        for b in range(n_beams):
            # accumulate frequency-domain summed spectrum for beam b
            summed = np.zeros((nf,), dtype=np.complex64)
            dvec = delays[b, :]
            
            for s in range(n_stations):
                # dvec[s] is scalar -> phase shape is (nf,)
                phase_s = np.exp(-2j * np.pi * (freqs * float(dvec[s])))
                summed += phase_s * X[s]
            
            # inverse FFT -> time domain beam
            beam_trace = np.fft.irfft(summed, n=nfft)[:n_samples].astype(np.float32)
            beam_list.append(beam_trace)
            
            # Log progress
            if n_beams > 50 and (b + 1) % 50 == 0:
                logger.info(f"CPU beamforming: {b+1}/{n_beams} beams completed")
        
        # Stack all beams - this may cause OOM if n_beams is very large
        try:
            beams = np.array(beam_list, dtype=np.float32)
        except MemoryError as e:
            logger.error(
                f"Out of memory stacking {n_beams} beams × {n_samples} samples. "
                f"Need {n_beams * n_samples * 4 / (1024**3):.2f} GB. "
                f"Consider reducing time window or processing fewer beams per call."
            )
            raise
        
        return beams, freqs
    
def compute_envelope(data, use_gpu: bool = True):
    """Compute a simple envelope for beams using absolute value.

    This function supports both NumPy and CuPy arrays and returns an
    array of the same shape as the input.
    """
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np

    arr = data
    if xp is cp and not isinstance(arr, cp.ndarray):
        arr = cp.asarray(arr)

    return xp.abs(arr)


def sta_lta(env, sampling_rate, sta_window, lta_window, use_gpu: bool = True):
    """Compute STA/LTA ratio on an envelope array.

    Parameters
    ----------
    env : ndarray
        Envelope (n_beams, n_samples) or (n_samples,)
    sampling_rate : float
        Sampling rate in Hz
    sta_window : float
        Short-term window (seconds)
    lta_window : float
        Long-term window (seconds)

    Returns
    -------
    ratio : ndarray
        STA/LTA ratio with same shape as input env
    """
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np

    arr = env
    scalar_input = False
    if arr.ndim == 1:
        arr = arr[None, :]
        scalar_input = True

    if xp is cp and not isinstance(arr, cp.ndarray):
        arr = cp.asarray(arr)
    else:
        arr = arr.astype(np.float64)

    n_beams, n_samples = arr.shape

    sta_samples = max(1, int(round(sta_window * sampling_rate)))
    lta_samples = max(1, int(round(lta_window * sampling_rate)))

    # operate on the provided envelope (should be non-negative)
    work = arr

    # cumsum with a leading zero to allow efficient trailing-window sums
    csum = xp.concatenate([xp.zeros((n_beams, 1), dtype=work.dtype), xp.cumsum(work, axis=1)], axis=1)

    def trailing_mean(window):
        if window <= 0:
            return xp.zeros((n_beams, n_samples), dtype=work.dtype)

        sums = csum[:, window:] - csum[:, :-window]
        if sums.shape[1] < n_samples:
            pad_width = n_samples - sums.shape[1]
            pad = xp.zeros((n_beams, pad_width), dtype=sums.dtype)
            means = xp.concatenate([pad, sums / float(window)], axis=1)
        else:
            means = sums / float(window)
        return means

    sta_full = trailing_mean(sta_samples)
    lta_full = trailing_mean(lta_samples)

    ratio = sta_full / (lta_full + 1e-12)

    if scalar_input:
        return ratio[0]
    return ratio


def classic_sta_lta(beams, sta_samples, lta_samples, use_gpu=True):
    """GPU/CPU implementation of ObsPy's classic_sta_lta for 2D beams.

    Computes STA and LTA as trailing averages of squared signal (energy)
    and returns an array of STA/LTA ratios with the same sample length as
    input beams. Works with numpy arrays or CuPy arrays.
    """
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np

    arr = beams
    if xp is cp and not isinstance(arr, cp.ndarray):
        arr = cp.asarray(arr)

    n_beams, n_samples = arr.shape

    # square the signal (energy) as in classic algorithm
    energy = arr * arr

    # cumulative sum with leading zero
    csum = xp.concatenate([xp.zeros((n_beams, 1), dtype=energy.dtype), xp.cumsum(energy, axis=1)], axis=1)

    def trailing_mean(window):
        if window <= 0:
            return xp.zeros((n_beams, n_samples), dtype=energy.dtype)

        sums = csum[:, window:] - csum[:, :-window]
        if sums.shape[1] < n_samples:
            pad_width = n_samples - sums.shape[1]
            pad = xp.zeros((n_beams, pad_width), dtype=sums.dtype)
            means = xp.concatenate([pad, sums / float(window)], axis=1)
        else:
            means = sums / float(window)
        return means

    sta_full = trailing_mean(sta_samples)
    lta_full = trailing_mean(lta_samples)

    # avoid divide-by-zero
    ratio = sta_full / (lta_full + 1e-12)
    return ratio


def classic_sta_lta(beams, nsta, nlta, use_gpu=True):
    """GPU/CPU implementation of ObsPy's classic_sta_lta for 2D arrays.

    beams : 2D array (n_beams, n_samples) or 1D (n_samples,) -- works for both
    nsta, nlta : int
        STA and LTA window lengths in samples
    Returns array of same shape (n_beams, n_samples) with STA/LTA ratio computed
    using trailing-window averages on the squared signal (matching ObsPy's
    general behaviour).
    """
    xp = cp if (use_gpu and CUPY_AVAILABLE) else np

    arr = beams
    # normalize shapes to 2D (n_beams, n_samples)
    scalar_input = False
    if arr.ndim == 1:
        arr = arr[None, :]
        scalar_input = True

    if xp is cp and not isinstance(arr, cp.ndarray):
        arr = cp.asarray(arr, dtype=cp.float64)
    else:
        arr = arr.astype(np.float64)

    if nsta <= 0 or nlta <= 0:
        raise ValueError("nsta and nlta must be positive integers")

    n_beams, n_samples = arr.shape

    # work on squared signal (energy)
    work = arr * arr

    # cumulative sum with leading zero
    csum = xp.concatenate([xp.zeros((n_beams, 1), dtype=work.dtype), xp.cumsum(work, axis=1)], axis=1)

    def trailing_mean(window):
        if window <= 0:
            return xp.zeros((n_beams, n_samples), dtype=work.dtype)

        sums = csum[:, window:] - csum[:, :-window]  # shape (n_beams, n_samples - window + 1)
        if sums.shape[1] < n_samples:
            pad_width = n_samples - sums.shape[1]
            pad = xp.zeros((n_beams, pad_width), dtype=sums.dtype)
            means = xp.concatenate([pad, sums / float(window)], axis=1)
        else:
            means = sums / float(window)
        return means

    sta_full = trailing_mean(nsta)
    lta_full = trailing_mean(nlta)

    # avoid division by zero
    ratio = sta_full / (lta_full + 1e-12)

    # ObsPy's implementation produces zero until the long-term average is
    # available (i.e. until index >= nlta). Force early samples to 0 so we
    # match that behavior and avoid huge ratios due to near-zero LTA values.
    # ObsPy returns the first non-zero value at index nlta-1 (0-based), so
    # zero-out samples strictly before nlta-1.
    first_valid = max(0, nlta - 1)
    if ratio.shape[1] > first_valid:
        if xp is cp:
            ratio[:, :first_valid] = 0
        else:
            ratio[:, :first_valid] = 0.0

    if scalar_input:
        # return 1D array for 1D input
        if xp is cp:
            return ratio[0]
        else:
            return ratio[0]
    return ratio


def simple_cpu_reference(traces: np.ndarray, delays_samples: np.ndarray) -> np.ndarray:
    """A naive time-domain integer-shift CPU reference implementation.

    Parameters
    ----------
    traces : np.ndarray
        (n_stations, n_samples)
    delays_samples : np.ndarray
        (n_beams, n_stations) delays in integer samples

    Returns
    -------
    beams : np.ndarray
        (n_beams, n_samples)
    """
    traces = np.asarray(traces)
    n_stations, n_samples = traces.shape
    n_beams = delays_samples.shape[0]
    beams = np.zeros((n_beams, n_samples), dtype=traces.dtype)
    for b in range(n_beams):
        for s in range(n_stations):
            shift = int(delays_samples[b, s])
            if shift >= 0:
                arr = np.concatenate([np.zeros(shift), traces[s, :n_samples-shift]])
            else:
                sshift = -shift
                arr = np.concatenate([traces[s, sshift:], np.zeros(sshift)])
            beams[b] += arr
    return beams
