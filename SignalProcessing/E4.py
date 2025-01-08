from scipy.signal import welch, csd, freqs, butter, ellip, cheby1, filtfilt, freqz
from scipy.io import loadmat
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# # Load the data from ImpTube12.mat
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']
# fs = int(data['fs'][0, 0])  # Sampling frequency
#
# # Extract the two signals (input and output)
# input_signal = Mic12[:, 0]  # First microphone signal (assumed input)
# output_signal = Mic12[:, 1]  # Second microphone signal (assumed output)
#
# # Define NFFT and calculate PSDs and CPSD
# NFFT = 1024
# f, Pxx = welch(input_signal, fs=fs, window=np.hanning(NFFT), nperseg=NFFT, noverlap=int(NFFT * 0.5), scaling='density')
# _, Pyy = welch(output_signal, fs=fs, window=np.hanning(NFFT), nperseg=NFFT, noverlap=int(NFFT * 0.5), scaling='density')
# _, Pxy = csd(input_signal, output_signal, fs=fs, nperseg=NFFT, noverlap=int(NFFT * 0.5), window=np.hanning(NFFT))
#
# # Restrict frequency range to 0-2000 Hz
# freq_range = f <= 2000
# f = f[freq_range]
# Pxx = Pxx[freq_range]
# Pxy = Pxy[freq_range]
#
# # Compute H1 and H2 estimates
# H1 = Pxy / Pxx
# H2 = Pyy / Pxy.conj()

# # Compute magnitude and phase
# H1_mag = np.abs(H1)
# H1_phase = np.angle(H1, deg=True)
# H2_mag = np.abs(H2)
# H2_phase = np.angle(H2, deg=True)

# # Plot Magnitudes
# plt.figure(figsize=(10, 6))
# plt.plot(f, H1_mag, label='H1 Magnitude')
# plt.plot(f, H2_mag, label='H2 Magnitude', linestyle='--')
# plt.title('Frequency Response Function Magnitudes')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Plot Phases
# plt.figure(figsize=(10, 6))
# plt.plot(f, H1_phase, label='H1 Phase')
# plt.plot(f, H2_phase, label='H2 Phase', linestyle='--')
# plt.title('Frequency Response Function Phases')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (degrees)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Step 1: Generate the multi-tone signal
# f = 10  # Fundamental frequency
# fs = 1000  # Sampling frequency (Hz)
# t = np.arange(1, 64001) / 1000  # Time vector (sampling frequency 1000 Hz)
# x = np.sin(2 * np.pi * f * t) + 2 * np.sin(3 * 2 * np.pi * f * t) + 0.5 * np.sin(7 * 2 * np.pi * f * t)

# # Step 2: Filter design parameters

# cutoff = 40  # Cutoff frequency (Hz) to remove 70 Hz component
# order = 4  # Filter order

# # Butterworth filter
# b_butter, a_butter = butter(order, cutoff / (fs / 2), btype='low')
# x_butter = filtfilt(b_butter, a_butter, x)  # Apply the filter
#
# # Elliptic filter (with ripple)
# rp = 1  # Passband ripple (dB)
# rs = 40  # Stopband attenuation (dB)
# b_ellip, a_ellip = ellip(order, rp, rs, cutoff / (fs / 2), btype='low')
# x_ellip = filtfilt(b_ellip, a_ellip, x)  # Apply the filter
#
# # Chebyshev Type 1 filter (with passband ripple)
# rp = 1  # Passband ripple (dB)
# b_cheby, a_cheby = cheby1(order, rp, cutoff / (fs / 2), btype='low')
# x_cheby = filtfilt(b_cheby, a_cheby, x)  # Apply the filter
#
# # Plot the unfiltered and filtered time signals
# plt.figure(figsize=(12, 4.2))
# plt.plot(t, x, label='Unfiltered Signal', color='black')
# plt.plot(t, x_butter, label='Butterworth Filter', color='blue', linestyle='--', linewidth=1)
# plt.plot(t, x_ellip, label='Elliptic Filter', color='red', linestyle='-.', linewidth=1)
# plt.plot(t, x_cheby, label='Chebyshev Filter', color='green', linestyle=':', linewidth=1)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.xlim(0, 0.4)  # Show 1 second of the signal
# plt.legend()
# plt.tight_layout()
# plt.savefig('E4_P4_1.eps', format='eps')
# plt.show()
#
#
# def plot_spectrum(signal, fs, label):
#     N = len(signal)
#     freqs = np.fft.rfftfreq(N, 1 / fs)
#     spectrum = np.abs(np.fft.rfft(signal)) / N
#     plt.plot(freqs, 20 * np.log10(spectrum), label=label)
#
# # Plot the spectra
# plt.figure(figsize=(12, 6))
# plot_spectrum(x, fs, 'Original Signal')
# plot_spectrum(x_butter, fs, 'Butterworth Filtered')
# plot_spectrum(x_ellip, fs, 'Elliptic Filtered')
# plot_spectrum(x_cheby, fs, 'Chebyshev Filtered')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.xlim(0, 100)  # Focus on 0-100 Hz range
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('E4_P4_2.eps', format='eps')
# plt.show()

# # Band-pass filter design parameters
# order = 4  # Filter order
# low_cutoff = 25  # Lower cutoff frequency (Hz)
# high_cutoff = 35  # Upper cutoff frequency (Hz)
# nyquist = fs / 2  # Nyquist frequency
# low_norm = low_cutoff / nyquist  # Normalized lower cutoff
# high_norm = high_cutoff / nyquist  # Normalized upper cutoff
#
# # 1. Butterworth Filter
# b_butter, a_butter = butter(order, [low_norm, high_norm], btype='band')
# x_butter = filtfilt(b_butter, a_butter, x)
#
# # 2. Chebyshev Filter (Passband ripple of 1 dB)
# rp = 1  # Passband ripple (dB)
# b_cheby1, a_cheby1 = cheby1(order, rp, [low_norm, high_norm], btype='band')
# x_cheby1 = filtfilt(b_cheby1, a_cheby1, x)
#
# # 3. Elliptic Filter (Passband ripple of 1 dB, Stopband attenuation of 40 dB)
# rs = 40  # Stopband attenuation (dB)
# b_ellip, a_ellip = ellip(order, rp, rs, [low_norm, high_norm], btype='band')
# x_ellip = filtfilt(b_ellip, a_ellip, x)
#
# # Compute Power Spectral Density (PSD) of the original and filtered signals
# f, psd_original = welch(x, fs, nperseg=1024)
# f, psd_butter = welch(x_butter, fs, nperseg=1024)
# f, psd_cheby1 = welch(x_cheby1, fs, nperseg=1024)
# f, psd_ellip = welch(x_ellip, fs, nperseg=1024)
#
# # Plot PSD of original and filtered signals
# plt.figure(figsize=(12, 5))
# plt.plot(f, 10 * np.log10(psd_original), label='Original Signal', color='black', linewidth=1.5)
# plt.plot(f, 10 * np.log10(psd_butter), label='Butterworth', color='blue')
# plt.plot(f, 10 * np.log10(psd_cheby1), label='Chebyshev', color='green')
# plt.plot(f, 10 * np.log10(psd_ellip), label='Elliptic', color='red')
#
# plt.xlim(0, 100)  # Focus on 0-100 Hz
# plt.ylim(-120, 0)  # Set y-limits
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('E4_P4_4.eps', format='eps')
# plt.show()


# # Load the data from ImpTube12.mat
# data = loadmat('ImpTube12.mat')
# Mic12 = data['Mic12']
# fs = int(data['fs'][0, 0])  # Sampling frequency
#
# # Extract the two signals (input and output)
# input_signal = Mic12[:, 0]  # First microphone signal (assumed input)
# output_signal = Mic12[:, 1]  # Second microphone signal (assumed output)
#
# # Band-pass filter design parameters
# order = 4  # Filter order
# low_cutoff = 700  # Lower cutoff frequency (Hz)
# high_cutoff = 760  # Upper cutoff frequency (Hz)
# nyquist = fs / 2  # Nyquist frequency
# low_norm = low_cutoff / nyquist  # Normalized lower cutoff
# high_norm = high_cutoff / nyquist  # Normalized upper cutoff
#
# # 2. Chebyshev Filter (Passband ripple of 1 dB)
# rp = 1  # Passband ripple (dB)
# b_cheby1, a_cheby1 = cheby1(order, rp, [low_norm, high_norm], btype='band')
# input_signal_cheby1 = filtfilt(b_cheby1, a_cheby1, input_signal)
# output_signal_cheby1 = filtfilt(b_cheby1, a_cheby1, output_signal)
#
# # Compute Power Spectral Density (PSD) of the original and filtered signals
# f, psd_input = welch(input_signal, fs, nperseg=1024)
# _, psd_output = welch(output_signal, fs, nperseg=1024)
# _, psd_input_cheby1 = welch(input_signal_cheby1, fs, nperseg=1024)
# _, psd_output_cheby1 = welch(output_signal_cheby1, fs, nperseg=1024)
#
# # Plot PSD of original and filtered signals
# plt.figure(figsize=(12, 5))
# plt.plot(f, 10 * np.log10(psd_input), label='Mic 1', color='blue', linewidth=1.5)
# plt.plot(f, 10 * np.log10(psd_output), label='Mic 2', color='black', linewidth=1.5)
# plt.plot(f, 10 * np.log10(psd_input_cheby1), label='Mic 1 Chebyshev', color='red', linewidth=1, linestyle='--')
# plt.plot(f, 10 * np.log10(psd_output_cheby1), label='Mic 2 Chebyshev', color='green', linewidth=1, linestyle='--')
#
# # plt.xlim(650, 770)  # Focus on 700-760 Hz range
# # plt.ylim(-100, 0)  # Set y-limits for better visualization
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (dB/Hz)')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('E4_P4_5.eps', format='eps')
# plt.show()


# # Step 1: Load the data from Heart_Noise.mat
# data = loadmat('Heart_Noise.mat')
# z = data['z'].squeeze()  # Extract and flatten the signal array
# t = data['t'].squeeze()  # Time vector
# fs = int(data['FS'].squeeze())  # Sampling frequency (convert to int)
# print(fs)
#
# # Step 2: Calculate the RMS Spectrum
# N = len(z)  # Length of the signal
# frequencies = np.fft.rfftfreq(N, 1 / fs)  # Frequency vector (only positive frequencies)
# fft_values = np.fft.rfft(z)  # Compute the FFT of the signal (only positive half of spectrum)
# rms_spectrum = np.abs(fft_values) / np.sqrt(N)  # RMS spectrum calculation
#
# # Step 3: Plot the RMS Spectrum
# plt.figure(figsize=(12, 4))
# plt.plot(frequencies, rms_spectrum, color='blue', linewidth=1.5)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('RMS Amplitude')
# plt.xlim(0, 100)  # Focus on the 0-100 Hz range to spot the 50 Hz disturbance
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.axvline(50, color='red', linestyle='--', label='50 Hz Disturbance')
# plt.legend()
# plt.tight_layout()
# plt.savefig('E4_P4_6.eps', format='eps')
# plt.show()
#
# # Step 2: Design a Butterworth Band-Stop (Notch) Filter
# low_cut = 48  # Lower cutoff frequency
# high_cut = 52  # Upper cutoff frequency
# order = 2  # Filter order (can be adjusted)
# nyquist = fs / 2  # Nyquist frequency (half the sampling rate)
# low = low_cut / nyquist  # Normalize to Nyquist (0-1 scale)
# high = high_cut / nyquist  # Normalize to Nyquist (0-1 scale)
#
# # Design the Butterworth Band-Stop Filter
# b, a = butter(order, [low, high], btype='bandstop')
#
# # Plot the frequency response of the filter
# w, h = freqz(b, a, worN=8000, fs=fs)
#
# # Step 3: Apply the Band-Stop Filter to the Noisy ECG Signal
# filtered_signal = filtfilt(b, a, z)  # Zero-phase filtering
#
# # Step 4: Plot the Original and Filtered Signals
#
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
#
# axs[0].plot(t, z, color='blue', label='Original signal', alpha=0.8, linewidth=1.5)
# axs[0].plot(t, filtered_signal, color='red', label='Filtered signal', linewidth=1.2)
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('Amplitude')
# axs[0].legend()
#
# # Optional: Plot the RMS spectrum of the filtered signal
# N = len(filtered_signal)
# frequencies = np.fft.rfftfreq(N, 1 / fs)
# fft_values = np.fft.rfft(filtered_signal)
# rms_spectrum = np.abs(fft_values) / np.sqrt(N)
#
# # Step 2: Calculate the RMS Spectrum
# N_orig = len(z)  # Length of the signal
# frequencies_orig = np.fft.rfftfreq(N_orig, 1 / fs)  # Frequency vector (only positive frequencies)
# fft_values_orig = np.fft.rfft(z)  # Compute the FFT of the signal (only positive half of spectrum)
# rms_spectrum_orig = np.abs(fft_values_orig) / np.sqrt(N_orig)  # RMS spectrum calculation
#
# axs[1].plot(frequencies, rms_spectrum, color='red', linewidth=1.5, label='Band stop filtered signal')
# axs[1].plot(frequencies_orig, rms_spectrum_orig, color='blue', linewidth=1,linestyle='--', label='Original signal')
# axs[1].set_xlabel('Frequency (Hz)')
# axs[1].set_ylabel('RMS Amplitude')
# axs[1].set_xlim(0, 100)  # Focus on the 0-100 Hz range to verify that 50 Hz is gone
# axs[1].set_ylim(-1, 11)  # Focus on the 0-100 Hz range to verify that 50 Hz is gone
# axs[1].axvline(50, color='red', linestyle='--', label='50 Hz Disturbance')
# axs[1].legend()
#
# plt.tight_layout()
# plt.savefig('E4_P4_7.eps', format='eps')
# plt.show()


