"""
Trace alignment and semblance for delay-and-sum array processing.

Semblance is the fraction of the array's energy that adds coherently on a beam:

    S = sum_t (sum_i a_i(t))**2 / (N * sum_t sum_i a_i(t)**2)

where ``a_i`` are the station traces after alignment. By Cauchy-Schwarz it lies
in [0, 1]: 1 for identical aligned traces, ~1/N for incoherent noise.

Pitfall worth knowing
---------------------
Do NOT build a grid-wide semblance by delay-and-summing the *squared* traces
(to get the denominator for every beam at once). Squared traces are not
band-limited, so a fractional frequency-domain shift makes them ring negative,
the denominator collapses, values clip to 1 and ``argmax`` lands on an
arbitrary grid cell. Maximise beam power over the grid -- the conventional
objective -- and evaluate semblance only at the winning cell, on traces aligned
by shifting the *band-limited waveforms*, as :func:`semblance_at` does.
"""

from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def align_traces(traces: np.ndarray, delays_sec: np.ndarray,
                 sampling_rate: float) -> np.ndarray:
    """Delay each station by ``delays_sec[i]`` with a frequency-domain shift.

    Uses the same convention as :func:`beam.core.gpu_beam.beamform_freq_domain`
    (station spectrum multiplied by ``exp(-2j*pi*f*tau)``), so delays from
    ``compute_delays`` align a plane wave from the corresponding direction.

    The traces are zero-padded to at least twice their length so the shift is
    not circular within the returned window. The returned array has the padded
    length: cropping back would discard energy shifted past the window edge and
    the semblance would no longer be bounded by 1.

    Parameters
    ----------
    traces : ndarray, shape (n_stations, n_samples)
    delays_sec : ndarray, shape (n_stations,)
    sampling_rate : float

    Returns
    -------
    aligned : ndarray, shape (n_stations, n_fft)
    """
    traces = np.asarray(traces, dtype=np.float64)
    delays_sec = np.asarray(delays_sec, dtype=np.float64)
    if traces.ndim != 2 or delays_sec.shape != (traces.shape[0],):
        raise ValueError("traces must be (n_stations, n_samples) and delays (n_stations,)")
    nfft = _next_pow2(2 * traces.shape[1])
    X = np.fft.rfft(traces, n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate)
    X *= np.exp(-2j * np.pi * f[None, :] * delays_sec[:, None])
    return np.fft.irfft(X, n=nfft, axis=1)


def semblance(aligned: np.ndarray) -> float:
    """Semblance of already-aligned traces, shape (n_stations, n_samples)."""
    a = np.asarray(aligned, dtype=np.float64)
    n = a.shape[0]
    den = n * np.sum(a * a)
    if n == 0 or den <= 0.0:
        return float("nan")
    return float(np.sum(np.sum(a, axis=0) ** 2) / den)


def semblance_at(traces: np.ndarray, delays_sec: np.ndarray,
                 sampling_rate: float) -> float:
    """Semblance of ``traces`` on the beam defined by ``delays_sec``."""
    return semblance(align_traces(traces, delays_sec, sampling_rate))
