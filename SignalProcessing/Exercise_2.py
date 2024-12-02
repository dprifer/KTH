import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
from scipy.fft import fft, fftfreq
import sounddevice as sd
from scipy.signal import spectrogram

# # Read the WAV file
# sample_rate, audio_data = wavfile.read('Sound4.wav')
# audio_data = audio_data[:, 0]
# audio_duration = len(audio_data) / sample_rate
# time = np.linspace(0, audio_duration, len(audio_data))
#
# fig, axs = plt.subplots(2, 1, figsize=(12, 6))
#
# axs[0].plot(time, audio_data, label="Original Signal", alpha=0.7)
# axs[0].set_title("Original Audio Signal in Time Domain")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Amplitude")
#
# # Listen to the original audio
# # print("Playing original audio:")
# # sd.play(audio_data, samplerate=sample_rate)
# # sd.wait()
#
# # Step 2: Resample the signal
# downsample_4 = resample(audio_data, len(audio_data) // 4)
# downsample_128 = resample(audio_data, len(audio_data) // 128)
#
# time_4 = np.linspace(0, audio_duration, len(downsample_4))
# time_128 = np.linspace(0, audio_duration, len(downsample_128))
#
# axs[1].plot(time, audio_data, label="Original Signal", alpha=0.7)
# axs[1].plot(time_4, downsample_4, label="Resampled (4th point)", alpha=0.7)
# axs[1].plot(time_128, downsample_128, label="Resampled (128th point)", alpha=0.7)
# axs[1].set_title("Comparison of Original and Resampled Signals")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Amplitude")
# axs[1].legend()
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("E2_P2_1.eps", format="eps", dpi=300)
#
#
# original_fs = sample_rate
#
# # Step 1: Define FFT size
# N = 512  # Number of FFT points
#
# # Step 2: Compute FFTs and one-sided spectra
# def compute_one_sided_spectrum(signal, fs, N):
#     # Compute FFT
#     fft_result = fft(signal[:N])
#     magnitude = np.abs(fft_result)[:N // 2]  # Take one-sided spectrum
#     freqs = np.linspace(0, fs / 2, N // 2)  # Frequency range for one-sided spectrum
#     magnitude[1:] *= 2  # Double the amplitude for all components except DC (0 Hz)
#     return freqs, magnitude
#
# # Compute spectra for original and resampled signals
# freqs_original, spectrum_original = compute_one_sided_spectrum(audio_data, original_fs, N)
# freqs_4, spectrum_4 = compute_one_sided_spectrum(downsample_4, original_fs / 4, N)
# freqs_128, spectrum_128 = compute_one_sided_spectrum(downsample_128, original_fs / 128, N)
#
# # Normalize the spectra for fair amplitude comparison
# spectrum_original /= max(spectrum_original)
# spectrum_4 /= max(spectrum_4)
# spectrum_128 /= max(spectrum_128)
#
# # Step 3: Plot the spectra
#
# fig, axs = plt.subplots(3, 1, figsize=(12, 8))
#
# axs[0].stem(freqs_original, spectrum_original, label="Original Sampling Rate")
# axs[0].set_title("One-sided spectra of original signal")
# axs[0].set_xlabel("Frequency (Hz)")
# axs[0].set_ylabel("Normalized Amplitude")
# axs[1].stem(freqs_4, spectrum_4, label="Resampled (4th Point)")
# axs[1].set_title("Resampled every 4th point")
# axs[1].set_xlabel("Frequency (Hz)")
# axs[1].set_ylabel("Normalized Amplitude")
# axs[2].stem(freqs_128, spectrum_128, label="Resampled (128th Point)")
# axs[2].set_title("Resampled every 128th point")
# axs[2].set_xlabel("Frequency (Hz)")
# axs[2].set_ylabel("Normalized Amplitude")
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("E2_P2_2.eps", format="eps", dpi=300)
#
# # Step 4: Compute and compare frequency band widths (∆f)
# delta_f_original = original_fs / N
# delta_f_4 = (original_fs / 4) / N
# delta_f_128 = (original_fs / 128) / N
#
# print(f"Original ∆f: {delta_f_original} Hz")
# print(f"Resampled (4th Point) ∆f: {delta_f_4} Hz")
# print(f"Resampled (128th Point) ∆f: {delta_f_128} Hz")


########################################################
## Aliasing
###########

# # Parameters
# fs = [32, 20, 16, 8]  # Sampling frequency (Hz)
# f = 10   # Signal frequency (Hz)
# duration = 1  # Duration of the signal (seconds)
#
# fig, axs = plt.subplots(4, 1, figsize=(12, 8))
#
# for i, fs in enumerate(fs):
#     # Time vector
#     t = np.arange(0, duration, 1 / fs)
#
#     # Create the sine wave signal
#     signal = np.sin(2 * np.pi * f * t)
#
#     # Perform FFT
#     N = len(t)  # Number of samples
#     yf = fft(signal)  # Fourier transform of the signal
#     xf = fftfreq(N, 1 / fs)  # Frequency bins
#
#     # Adjust the amplitude for the one-sided spectrum
#     yf = 2.0 * np.abs(yf)  # Scaling factor for one-sided spectrum
#
#     # Only take the positive half of the spectrum
#     xf = xf[:N // 2]
#     yf = yf[:N // 2]
#
#     axs[i].stem(xf, np.abs(yf))
#     axs[i].set_title(f"One-sided spectra with {fs} Hz sampling")
#     axs[i].set_xlabel("Frequency (Hz)")
#     axs[i].set_ylabel("Amplitude")
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("E2_P2_3.eps", format="eps", dpi=300)



########################################################
## Spectogram
#############

# Step 1: Load the audio file
filename = "A320_232.wav"  # Replace with your file name
samplerate, data = wavfile.read(filename)
audio_duration = len(data) / samplerate
time = np.linspace(0, audio_duration, len(data))

# print("Playing original audio:")
# sd.play(data, samplerate=samplerate)
# sd.wait()

# fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
#
# f1, t1, Sxx1 = spectrogram(data, fs=samplerate, window='hann', nperseg=512, noverlap=512//2, scaling='density')
# pcm1 = axs[0].pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud')
#
# f2, t2, Sxx2 = spectrogram(data, fs=samplerate, window='hann', nperseg=8192, noverlap=8192//2, scaling='density')
# pcm2 = axs[1].pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud')
#
# f3, t3, Sxx3 = spectrogram(data, fs=samplerate, window='hann', nperseg=2097152, noverlap=2097152//2, scaling='density')
# pcm3 = axs[2].pcolormesh(t3, f3, 10 * np.log10(Sxx3), shading='gouraud')
#
# fig.subplots_adjust(right=0.85)
#
# # Add a common colorbar
# cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(pcm1, cax=cbar_ax, label='Power/Frequency (dB/Hz)')
#
# # Set subplot titles and labels
# axs[0].set_title('Spectrogram with NFFT=512')
# axs[0].set_ylabel('Frequency [Hz]')
# axs[1].set_title('Spectrogram with NFFT=8192')
# axs[2].set_title('Spectrogram with NFFT=2097152')
# for ax in axs:
#     ax.set_xlabel('Time [s]')
#     # ax.set_xlim(25, 30)
#     # ax.set_ylim(500, 750)
#
# # Show the plot
# plt.show()
# fig.savefig("E2_P2_4.png", dpi=300)

# f, t, Sxx = spectrogram(data, fs=samplerate, window='hann', nperseg=8192, noverlap=8192//2, scaling='density')
#
# fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharey=True)
# pcolormesh = axs.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
# axs.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
#
# # Set subplot titles and labels
# axs.set_ylabel('Frequency [Hz]')
# axs.set_xlabel('Time [s]')
# fig.colorbar(pcolormesh, ax=axs, label='Power [dB]')
#
# plt.show()
# fig.savefig("E2_P2_5.png", dpi=300)

v_sound = 343  # m/s
f_source = 500  # Hz
f_detected = 390  # Hz

v_plane = v_sound * (1-f_source/f_detected) * (2**0.5)
print(v_plane)
v_kmh = v_plane*3.6
print(v_kmh)

