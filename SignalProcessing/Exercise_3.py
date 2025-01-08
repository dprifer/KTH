import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, rfftfreq, fftfreq
# from scipy.signal.windows import hann
from scipy.signal import csd, welch, windows
from scipy.io import loadmat

# # Parameters
# f = 10  # Frequency in Hz
# fs = 1000
# t = np.arange(1, 64001) / fs  # Time vector in seconds, sampling rate of 1000 Hz
# x = np.sin(2 * np.pi * f * t) + np.random.randn(len(t)) / 10  # Signal with noise
#
# # Extract the first 400 samples
# x_part_400 = x[:400]
# t_part_400 = t[:400]
#
# x_part_450 = x[:450]
# t_part_450 = t[:450]
#
# w1 = hann(400, sym=False)  # Hanning window for 400 samples
# w2 = hann(450, sym=False)  # Hanning window for 450 samples
#
# x_part_400_windowed = x_part_400 * w1
# x_part_450_windowed = x_part_450 * w2
#
# # Compute the FFT
# fft_x_400 = rfft(x_part_400) / float(400/2)
# fft_x_400_windowed = rfft(x_part_400_windowed) / float(400/2)
# fft_x_400_windowed_corr = rfft(x_part_400_windowed) / float(400/2) / 0.5
# frequencies_400 = rfftfreq(400, 1./fs)
#
# fft_x_450 = rfft(x_part_450) / float(450/2)
# fft_x_450_windowed = rfft(x_part_450_windowed) / float(450/2)
# fft_x_450_windowed_corr = rfft(x_part_450_windowed) / float(450/2) / 0.5
# frequencies_450 = rfftfreq(450, 1./fs)
#
# # Plot the signal versus time
# plt.figure(figsize=(12, 6))
#
# # Time-domain plot
# plt.subplot(2, 1, 1)
# plt.plot(frequencies_400, abs(fft_x_400), label="Without windowing")
# plt.plot(frequencies_400, abs(fft_x_400_windowed), label="With Hanning window")
# plt.plot(frequencies_400, abs(fft_x_400_windowed_corr), label="With Hanning window")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title("400 samples (periodic signal portion)")
# plt.legend()
# plt.grid()
#
# # Frequency-domain plot
# plt.subplot(2, 1, 2)
# plt.plot(frequencies_450, abs(fft_x_450), label="Without windowing")
# plt.plot(frequencies_450, abs(fft_x_450_windowed), label="With Hanning window")
# plt.plot(frequencies_450, abs(fft_x_450_windowed_corr), label="With Hanning window")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title("450 samples (non-periodic signal portion)")
# plt.legend()
# plt.grid()
#
# plt.tight_layout()
# # plt.savefig("E3_P1_5.eps", format="eps")
# plt.show()


# Welch method
##############

# # Load data from ImpTube12.mat
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']  # Two-column matrix
# fs = int(data['fs'][0][0])  # Sampling frequency (scalar)
#
# # Extract the two signals
# signal1 = Mic12[:, 0]
# signal2 = Mic12[:, 1]
#
# # Define Welch parameters
# nfft_values = [1024, 7680]  # Block sizes
# overlap_factor = 0.5  # 50% overlap

# Problem 1
# # Calculate and plot PSDs for each signal
# plt.figure(figsize=(12, 6))
# for i in range(2):
#     plt.subplot(2, 1, i+1)
#     for signal_idx, signal in enumerate([signal1, signal2], start=1):
#         for nfft in nfft_values:
#             window = np.hanning(nfft)  # Hanning window
#             overlap = int(nfft * overlap_factor)  # 50% overlap
#
#             # Compute the Welch PSD
#             f, psd = welch(signal, fs=fs, window=window, nperseg=nfft, noverlap=overlap, scaling='density')
#
#             # Plot PSD
#             plt.semilogy(f, psd, label=f'Signal {signal_idx}, NFFT = {nfft}')
#
#
#     # Formatting the plot
#     plt.title(f"Power Spectral Density")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power Spectral Density (dB/Hz)")
#     plt.legend()
#     plt.grid()
# plt.tight_layout()
# plt.savefig(f'E3_P2_1.eps', format='eps')  # Save the figure in EPS format
# plt.show()

# Problem 2
# L = len(signal1)  # Signal length
# print(L)
#
# # plt.figure(figsize=(12, 6))
# plt.figure(figsize=(14, 6))
# i=0
# for nfft in nfft_values:
#     # plt.subplot(2, 1, i)
#
#     window = np.hanning(nfft)  # Hanning window
#     overlap = int(nfft * overlap_factor)
#
#     # Calculate cross spectral densities
#     f12, S12 = csd(signal1, signal2, fs=fs, window=window, nperseg=nfft, noverlap=overlap)
#     f21, S21 = csd(signal2, signal1, fs=fs, window=window, nperseg=nfft, noverlap=overlap)
#
#     # # Plot cross spectral densities
#     # plt.semilogy(f12, np.abs(S12), label=f'S12')
#     # plt.semilogy(f21, np.abs(S21), linestyle='--', label='S21')
#     # plt.title(f"Cross Spectral Densities for NFFT={nfft}")
#     # plt.xlabel("Frequency (Hz)")
#     # plt.ylabel("CPSD")
#     # plt.legend()
#     # plt.grid()
#     #
#     # # Calculate the number of averages
#     # num_averages = int(L / nfft * 2 - 1)
#     # print(f"NFFT = {nfft}: Number of averages = {num_averages}")
#
#     # Calculate amplitude and phase
#     amplitude12 = np.abs(S12)
#     amplitude21 = np.abs(S21)
#     phase12 = np.angle(S12)
#     phase21 = np.angle(S21)
#
#     # Plot amplitude and phase
#     # Amplitude plot
#     plt.subplot(2, 2, 1+i)
#     plt.plot(f12, amplitude12, label='S12')
#     plt.plot(f21, amplitude21, label='S21')
#     plt.title(f"Amplitude of CPSD (NFFT={nfft})")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid()
#
#     # Phase plot
#     plt.subplot(2, 2, 3+i)
#     plt.plot(f12, phase12, label='S12')
#     plt.plot(f21, phase21, label='S21')
#     plt.title(f"Phase of CPSD (NFFT={nfft})")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Phase (radians)")
#     plt.legend()
#     plt.grid()
#
#     i += 1
#
# # plt.tight_layout()
# # plt.savefig(f'E3_P2_2.eps', format='eps')
# # plt.show()
#
# plt.tight_layout()
# plt.savefig('E3_P2_3.eps', format='eps')
# plt.show()
#
# window = np.hanning(1024)  # Hanning window
# overlap = int(1024 * overlap_factor)
# f12, S12 = csd(signal1, signal2, fs=fs, window=window, nperseg=1024, noverlap=overlap)
#
# plt.figure(figsize=(14, 6))
# plt.semilogy(f12, S12.real, label="Real Part")
# plt.semilogy(f12, S12.imag, label="Imaginary Part")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('E3_P2_4.eps', format='eps')
# plt.show()


# # Problem 2.2
# #############
#
# # Load data
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']
# fs = int(data['fs'][0][0])
#
# # Extract one signal
# signal = Mic12[:, 0]
#
# # Parameters
# nfft = 1024
# overlap = int(0.5 * nfft)  # 50% overlap
# window_hanning = np.hanning(nfft)
# window_rect = np.ones(nfft)
#
# print(fs)
#
# # Manual Welch PSD Calculation
# def welch_psd(signal, fs, window, nfft, overlap):
#     segment_count = 0
#     psd_accum = np.zeros(nfft // 2 + 1)  # Accumulate PSD
#
#     step_size = nfft - overlap
#
#     for start in range(0, len(signal) - nfft + 1, step_size):
#         segment = signal[start:start + nfft]
#         windowed = segment * window
#
#         # FFT and scaling
#         fft_result = np.fft.rfft(windowed, n=nfft)
#         power = 2 * np.abs(fft_result) ** 2 / (np.sum(window ** 2) * fs)
#         power[0] = (np.abs(fft_result[0]) ** 2) / (np.sum(window ** 2) * fs)
#         power[-1] = (np.abs(fft_result[-1]) ** 2) / (np.sum(window ** 2) * fs)
#
#         psd_accum += power
#         segment_count += 1
#
#     # Average the accumulated PSD
#     psd_avg = psd_accum / segment_count
#     freqs = np.fft.rfftfreq(nfft, 1 / fs)
#     return freqs, psd_avg
#
# # Calculate PSD manually with Hanning window
# freqs, psd_manual_hanning = welch_psd(signal, fs, window_hanning, nfft, overlap)
#
# # Calculate PSD manually with rectangular window
# _, psd_manual_rect = welch_psd(signal, fs, window_rect, nfft, overlap)
#
# # Calculate PSD using built-in Welch function
# f, psd = welch(signal, fs=fs, window=window_hanning, nperseg=nfft, noverlap=overlap, scaling='density')
#
# # Plotting Results
# plt.figure(figsize=(14, 6))
#
# # Plot comparison: Manual vs Welch (Hanning)
# plt.subplot(1, 2, 1)
# plt.semilogy(freqs, psd_manual_hanning, label="Manual PSD (Hanning)")
# plt.semilogy(f, psd, '--', label="Built-in Welch PSD")
# plt.title("Manual vs SciPy Welch")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD")
# plt.legend()
# plt.grid()
#
# # Plot comparison: Hanning vs Rectangular
# plt.subplot(1, 2, 2)
# plt.semilogy(freqs, psd_manual_hanning, label="Manual PSD (Hanning)")
# plt.semilogy(freqs, psd_manual_rect, '--', label="Manual PSD (Rectangular)")
# plt.title("Hanning vs rectangular windowing")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD")
# plt.legend()
# plt.grid()
#
# plt.tight_layout()
# plt.savefig('E3_P2_5.eps', format='eps')
# plt.show()
#


# # Part 3
# #########
#
# # Define the signal
# fs = 1000  # Sampling frequency (Hz)
# f = 10     # Frequency of sinusoid (Hz)
# t = np.arange(1, 64001) / fs  # Time vector
# x = np.sin(2 * np.pi * f * t) + np.random.randn(len(t)) / 10  # Noisy sinusoid
#
# # Block sizes
# NFFT_1 = 100
# NFFT_2 = 400
#
# # Perform FFT for NFFT = 100
# fft_result_100 = np.fft.rfft(x[:NFFT_1], NFFT_1)
# frequencies_100 = np.fft.rfftfreq(NFFT_1, d=1/fs)
# magnitude_spectrum_100 = np.abs(fft_result_100)
#
# # Perform FFT for NFFT = 400
# fft_result_400 = np.fft.rfft(x[:NFFT_2], NFFT_2)
# frequencies_400 = np.fft.rfftfreq(NFFT_2, d=1/fs)
# magnitude_spectrum_400 = np.abs(fft_result_400)
#
# # Plot the spectra
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(frequencies_100, magnitude_spectrum_100, label='NFFT = 100')
# plt.plot(frequencies_400, magnitude_spectrum_400, label='NFFT = 400')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Sinusoid signal')
# plt.legend()
# plt.grid()
#
# # Load the MATLAB .mat file
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']
# fs = int(data['fs'][0, 0])  # Sampling frequency
#
# # Select the first microphone signal (first column of Mic12)
# signal = Mic12[:, 0]
#
# # Block sizes
# NFFT_1 = 1024
# NFFT_2 = 8192
#
# # Perform FFT for NFFT = 1024
# fft_result_1024 = np.fft.rfft(signal[:NFFT_1], NFFT_1)
# frequencies_1024 = np.fft.rfftfreq(NFFT_1, d=1/fs)
# magnitude_spectrum_1024 = np.abs(fft_result_1024)
#
# # Perform FFT for NFFT = 8192
# fft_result_8192 = np.fft.rfft(signal[:NFFT_2], NFFT_2)
# frequencies_8192 = np.fft.rfftfreq(NFFT_2, d=1/fs)
# magnitude_spectrum_8192 = np.abs(fft_result_8192)
#
# # Plot the spectra
# plt.subplot(2, 1, 2)
# plt.plot(frequencies_8192, magnitude_spectrum_8192, label='NFFT = 8192')
# plt.plot(frequencies_1024, magnitude_spectrum_1024, label='NFFT = 1024')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Microphone signal')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig('E3_P3_1.eps', format='eps')
# plt.show()
#
# # Generate the sinusoidal signal with added noise
# f = 10  # Frequency in Hz
# t = np.arange(1, 64001) / 1000  # Time vector (1 ms sampling rate)
# x = np.sin(2 * np.pi * f * t) + np.random.randn(len(t)) / 10  # Signal with noise
#
# # Define the block sizes for PSD estimation
# NFFT_1 = 100
# NFFT_2 = 400
#
# # Calculate the PSD for NFFT = 100
# f_1, Pxx_1 = welch(x, fs=1000, window=np.hanning(NFFT_1), nperseg=NFFT_1, noverlap=int(NFFT_1 * 0.5), scaling='density')
#
# # Calculate the PSD for NFFT = 400
# f_2, Pxx_2 = welch(x, fs=1000, window=np.hanning(NFFT_2), nperseg=NFFT_2, noverlap=int(NFFT_2 * 0.5), scaling='density')
#
# # Plot the two spectra in the same figure
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.semilogy(f_1, Pxx_1, label='NFFT = 100')
# plt.semilogy(f_2, Pxx_2, label='NFFT = 400')
# plt.title('PSD of sinusoidal signal')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')
# plt.legend()
# plt.grid(True)
#
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']
# fs = int(data['fs'][0, 0])  # Sampling frequency
#
# # Select the first microphone signal (first column of Mic12)
# signal = Mic12[:, 0]
#
# # Define the block sizes for PSD estimation
# NFFT_1 = 1024
# NFFT_2 = 8192
#
# # Calculate the PSD for NFFT = 1024
# f_1, Pxx_1 = welch(signal, fs=fs, window=np.hanning(NFFT_1), nperseg=NFFT_1, noverlap=int(NFFT_1 * 0.5), scaling='density')
#
# # Calculate the PSD for NFFT = 8192
# f_2, Pxx_2 = welch(signal, fs=fs, window=np.hanning(NFFT_2), nperseg=NFFT_2, noverlap=int(NFFT_2 * 0.5), scaling='density')
#
# # Plot the two spectra in the same figure
# plt.subplot(2, 1, 2)
# plt.semilogy(f_1, Pxx_1, label='NFFT = 1024')
# plt.semilogy(f_2, Pxx_2, label='NFFT = 8192')
# plt.title('PSD of the microphone signal')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig('E3_P3_2.eps', format='eps')
# plt.show()
#



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, ellip, freqz

# Filter design parameters
fs = 1000  # Sampling frequency
order = 4  # Filter order
cutoff = 40  # Cutoff frequency (Hz)
nyquist = fs / 2  # Nyquist frequency

# Design different filters
# 1. Butterworth (No ripple)
b_butter, a_butter = butter(order, cutoff / nyquist, btype='low')
w_butter, h_butter = freqz(b_butter, a_butter, worN=8000)

# 2. Chebyshev Type 1 (Ripple in passband)
rp = 1  # Passband ripple (dB)
b_cheby1, a_cheby1 = cheby1(order, rp, cutoff / nyquist, btype='low')
w_cheby1, h_cheby1 = freqz(b_cheby1, a_cheby1, worN=8000)

# 3. Elliptic (Ripple in passband and stopband)
rp = 1  # Passband ripple (dB)
rs = 40  # Stopband attenuation (dB)
b_ellip, a_ellip = ellip(order, rp, rs, cutoff / nyquist, btype='low')
w_ellip, h_ellip = freqz(b_ellip, a_ellip, worN=8000)

# Plot the magnitude response (dB)
plt.figure(figsize=(12, 5))
plt.plot(w_butter * fs / (2 * np.pi), 20 * np.log10(abs(h_butter)), label='Butterworth (No Ripple)', color='blue')
plt.plot(w_cheby1 * fs / (2 * np.pi), 20 * np.log10(abs(h_cheby1)), label='Chebyshev Type I (Passband Ripple)', color='green')
plt.plot(w_ellip * fs / (2 * np.pi), 20 * np.log10(abs(h_ellip)), label='Elliptic (Passband + Stopband Ripple)', color='red')

# Draw a line at 0 dB (perfect passband) and -40 dB (stopband attenuation)
plt.axhline(-40, color='gray', linestyle='--', label='-40 dB (Stopband attenuation)')
plt.axhline(0, color='gray', linestyle='--', label='0 dB (Passband level)')

plt.xlim(0, 100)  # Focus on 0-100 Hz
plt.ylim(-60, 5)  # Set y-limits to focus on passband and stopband
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('E4_P4_3.eps', format='eps')
plt.show()