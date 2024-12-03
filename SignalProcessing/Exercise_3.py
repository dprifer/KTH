import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, rfftfreq, fftfreq
from scipy.signal.windows import hann

# Parameters
f = 10  # Frequency in Hz
fs = 1000
t = np.arange(1, 64001) / fs  # Time vector in seconds, sampling rate of 1000 Hz
x = np.sin(2 * np.pi * f * t) + np.random.randn(len(t)) / 10  # Signal with noise

# Extract the first 400 samples
x_part_400 = x[:400]
t_part_400 = t[:400]

x_part_450 = x[:450]
t_part_450 = t[:450]

w1 = hann(400, sym=False)  # Hanning window for 400 samples
w2 = hann(450, sym=False)  # Hanning window for 450 samples

x_part_400_windowed = x_part_400 * w1
x_part_450_windowed = x_part_450 * w2

# Compute the FFT
fft_x_400 = rfft(x_part_400) / float(400/2)
fft_x_400_windowed = rfft(x_part_400_windowed) / float(400/2)
fft_x_400_windowed_corr = rfft(x_part_400_windowed) / float(400/2) / 0.5
frequencies_400 = rfftfreq(400, 1./fs)

fft_x_450 = rfft(x_part_450) / float(450/2)
fft_x_450_windowed = rfft(x_part_450_windowed) / float(450/2)
fft_x_450_windowed_corr = rfft(x_part_450_windowed) / float(450/2) / 0.5
frequencies_450 = rfftfreq(450, 1./fs)

# Plot the signal versus time
plt.figure(figsize=(12, 6))

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(frequencies_400, abs(fft_x_400), label="Without windowing")
plt.plot(frequencies_400, abs(fft_x_400_windowed), label="With Hanning window")
plt.plot(frequencies_400, abs(fft_x_400_windowed_corr), label="With Hanning window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("400 samples (periodic signal portion)")
plt.legend()
plt.grid()

# Frequency-domain plot
plt.subplot(2, 1, 2)
plt.plot(frequencies_450, abs(fft_x_450), label="Without windowing")
plt.plot(frequencies_450, abs(fft_x_450_windowed), label="With Hanning window")
plt.plot(frequencies_450, abs(fft_x_450_windowed_corr), label="With Hanning window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("450 samples (non-periodic signal portion)")
plt.legend()
plt.grid()

plt.tight_layout()
# plt.savefig("E3_P1_5.eps", format="eps")
plt.show()