import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(accel, time, title="Acceleration time history"):
    plt.figure(figsize=(10, 4))
    plt.plot(time, accel, lw=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

def plot_spectrogram(stft_magnitude, freqs, time_frames, vmin=None, vmax=None,
                     title="STFT magnitude spectrogram"):
    plt.figure(figsize=(10, 4))
    mag_db = 10 * np.log10(stft_magnitude + 1e-12)
    if vmin is None:
        vmin = np.percentile(mag_db, 5)
    if vmax is None:
        vmax = np.percentile(mag_db, 95)
    plt.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[time_frames[0], time_frames[-1], freqs[0], freqs[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    plt.colorbar(label="Magnitude [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()

def plot_spectrogram_with_peaks(stft_magnitude, freqs, time_frames, peaks,
                                max_peaks=200, **kwargs):
    plot_spectrogram(stft_magnitude, freqs, time_frames, **kwargs)
    if peaks:
        # peaks: list of (freq_hz, time_sec, magnitude)
        subset = peaks[:max_peaks]
        t = [p[1] for p in subset]
        f = [p[0] for p in subset]
        plt.scatter(t, f, s=15, c="red", alpha=0.6, edgecolors="none", label="Peaks")
        plt.legend(loc="upper right")

def plot_constellation(peaks, fingerprints, max_pairs=500):
    plt.figure(figsize=(6, 4))
    if peaks:
        t = [p[1] for p in peaks]
        f = [p[0] for p in peaks]
        plt.scatter(t, f, s=10, c="lightgray", label="Peaks")
    # Optionally overlay a random subset of peak pairs from fingerprints
    pair_count = 0
    for (f1_bin, f2_bin, t_bin), times in fingerprints.items():
        if pair_count >= max_pairs:
            break
        # Here you’d map bins back to approx Hz/s if needed
        pair_count += 1
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Constellation map (peaks and selected pairs)")
    plt.tight_layout()