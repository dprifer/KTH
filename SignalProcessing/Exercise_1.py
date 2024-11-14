from cProfile import label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, wavfile
#from scipy.special import label
from scipy.stats import skew, kurtosis
from scipy.signal import correlate
import sounddevice as sd
import librosa
import os


current_dir = os.getcwd()


#######################################################################################################################
## Part 1_1
############

# # Audio file names
# file_names = ['Sound1.wav', 'Sound2.wav', 'Sound3.wav', 'Sound4.wav']
#
# # Get bit depth
# def get_bit_depth(audio_data):
#     # Check the dtype to infer the bit depth
#     if audio_data.dtype == np.int16:
#         return 16
#     elif audio_data.dtype == np.int32:
#         return 32
#     elif audio_data.dtype == np.float32:
#         # Assume 32-bit float audio, can be variable depth
#         return '32-bit (float)'
#     elif audio_data.dtype == np.float64:
#         # Assume 64-bit float audio, rare in common audio formats
#         return '64-bit (float)'
#     else:
#         return 'Unknown'
#
#
# fig, axs = plt.subplots(4, 1, figsize=(12, 10))
# for i, file_name in enumerate(file_names):
#
#     file_path = os.path.join(current_dir, file_name)
#
#     if not os.path.exists(file_name):
#         print(f"File '{file_name}' not found.")
#         continue
#
#     # Read the audio file
#     sampling_frequency, audio_data = wavfile.read(file_name)
#     print(sampling_frequency)
#
#     # Get the number of bits (bit depth)
#     bit_depth = get_bit_depth(audio_data)
#     print(bit_depth)
#
#     # Time axis for plotting
#     duration = len(audio_data) / sampling_frequency
#     time = np.linspace(0, duration, num=len(audio_data))
#
#     # Plot the audio signal
#     if audio_data.ndim == 1:  # Mono audio
#         axs[i].plot(time, audio_data, label='Mono')
#     else:  # Stereo audio
#         axs[i].plot(time, audio_data[:, 0], label='Left Channel', color='blue')
#         axs[i].plot(time, audio_data[:, 1], label='Right Channel', color='orange')
#
#     axs[i].set_title(f'{file_name} - Sampling Frequency: {sampling_frequency} Hz, Bit Depth: {bit_depth}')
#     axs[i].set_xlabel('Time [s]')
#     axs[i].set_ylabel('Amplitude')
#     axs[i].legend()
#     axs[i].grid(True)
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("P1_1.eps", format="eps", dpi=300)
#
#
# ######################################################################################################################
# # Part 1_2
# ###########
#
# sampling_frequency, audio_data = wavfile.read('Sound3.wav')
# duration = len(audio_data) / sampling_frequency
# time = np.linspace(0, duration, num=len(audio_data))
#
# # RMS value
# # RMS is equivalent to the standard deviation for a zero-mean signal
# mean_valueL = np.mean(audio_data[:, 0])
# mean_valueR = np.mean(audio_data[:, 1])
# rms_valueL = np.sqrt(np.mean((audio_data[:, 0] - mean_valueL) ** 2))
# rms_valueR = np.sqrt(np.mean((audio_data[:, 1] - mean_valueR) ** 2))
#
# # Step 3: Calculate Peak Value
# peak_valueL = np.max(np.abs(audio_data[:, 0]))
# peak_valueR = np.max(np.abs(audio_data[:, 1]))
#
# # Step 4: Calculate Peak-to-Peak Value
# peak_to_peak_valueL = np.ptp(audio_data[:, 0])
# peak_to_peak_valueR = np.ptp(audio_data[:, 1])
#
# # Step 5: Calculate Crest Factor
# crest_factorL = peak_valueL / rms_valueL
# crest_factorR = peak_valueR / rms_valueR
#
# # Step 6: Print the results
# print(f"Sampling Frequency: {sampling_frequency} Hz")
# print(f"RMS Value: L = {rms_valueL}, R = {rms_valueR}")
# print(f"Peak Value: L = {peak_valueL}, R = {peak_valueR}")
# print(f"Peak-to-Peak Value: L = {peak_to_peak_valueL}, R = {peak_to_peak_valueR}")
# print(f"Crest Factor: L = {crest_factorL}, R = {crest_factorR}")
#
# plt.figure(figsize=(10, 4))
# plt.plot(time, audio_data)
# plt.title(f'Audio Signal of Sound3.wav')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()
#
# table = {
#     'Property': ['RMS Value', 'Peak Value', 'Peak-to-Peak Value', 'Crest Factor'],
#     'Left Channel': [rms_valueL, peak_valueL, peak_to_peak_valueL, crest_factorL],
#     'Right Channel': [rms_valueR, peak_valueR, peak_to_peak_valueR, crest_factorR]
# }
#
# df = pd.DataFrame(table)
#
# # Convert DataFrame to LaTeX
# latex_code = df.to_latex(index=False, caption='Properties of Sound3.wav signal', label='tab:properties', escape=False, float_format="%.2f")
#
# # Modify LaTeX code to insert \centering after the first row
# # latex_code = latex_code.replace('\\begin{tabular}{lrr}', '\\begin{tabular}{lrr}\n\\centering')
#
# print(latex_code)
#
#
# ######################################################################################################################
# # Part 1_3
# ###########
#
# # Load the .mat file
# data = loadmat('Bearing.mat')
#
# # Assume the variable names in the .mat file are 'healthy_signal' and 'faulty_signal'
# bearing1_signal = data['Bearing1'].flatten()  # Flatten in case the signal is a 2D array
# bearing2_signal = data['Bearing2'].flatten()
#
# bearing1_skewness = skew(bearing1_signal)
# bearing1_kurtosis = kurtosis(bearing1_signal)
#
# bearing2_skewness = skew(bearing2_signal)
# bearing2_kurtosis = kurtosis(bearing2_signal)
#
# # Print skewness and kurtosis values
# print(f"Bearing 1 - Skewness: {bearing1_skewness:.4f}, Kurtosis: {bearing1_kurtosis:.4f}")
# print(f"Bearing 2 - Skewness: {bearing2_skewness:.4f}, Kurtosis: {bearing2_kurtosis:.4f}")
#
# # Plot the PDFs for both signals
# counts1, bin_edges1 = np.histogram(bearing1_signal, bins=20, density=True)
# counts2, bin_edges2 = np.histogram(bearing2_signal, bins=20, density=True)
# peak_bearing1 = counts1.max()
# peak_bearing2 = counts2.max()
# print(f"Peak of Bearing 1 PDF: {peak_bearing1:.4f}")
# print(f"Peak of Bearing 2 PDF: {peak_bearing2:.4f}")
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
# axs[0].plot(bearing1_signal, label='Bearing1')
# axs[0].plot(bearing2_signal, label='Bearing2')
# axs[0].set_xlabel('Sample')
# axs[0].set_ylabel('Signal Value')
# axs[0].legend()
# axs[1].bar(bin_edges1[:-1], counts1, width=np.diff(bin_edges1), edgecolor='black', alpha=0.7, label='Bearing1')
# axs[1].set_xlabel('Signal Value')
# axs[1].set_ylabel('Probability Density')
# axs[1].bar(bin_edges2[:-1], counts2, width=np.diff(bin_edges2), edgecolor='black', alpha=0.7, label='Bearing2')
#
# # Add skewness and kurtosis text for Bearing 1
# axs[1].text(0.05, 0.9, f'Skewness: {bearing1_skewness:.4f}\nKurtosis: {bearing1_kurtosis:.2f}',
#             transform=axs[1].transAxes, verticalalignment='top', color='blue')
#
# # Add skewness and kurtosis text for Bearing 2
# axs[1].text(0.05, 0.75, f'Skewness: {bearing2_skewness:.4f}\nKurtosis: {bearing2_kurtosis:.2f}',
#             transform=axs[1].transAxes, verticalalignment='top', color='orange')
#
# axs[1].legend()
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("P1_2.eps", format="eps", dpi=300)
#
#
#
# ######################################################################################################################
# # Convolution
# ################
#
# x1 = np.array([0, 1, 0, 0, 0, 0])
# x2 = np.array([0, 1, 0, 0, 1, 0])
# x3 = 2 * np.sin(np.arange(0, 2 * np.pi, 0.1))
#
# # Define system impulse response
# h = np.array([-1, 1, 3, 5, 3, 1, -1, -3])
#
# # Perform discrete convolution for each input
# y1 = np.convolve(x1, h, mode='full')
# y2 = np.convolve(x2, h, mode='full')
# y3 = np.convolve(x3, h, mode='full')
#
# # Plot outputs in the time domain
# fig, axs = plt.subplots(3, 1, figsize=(12, 6))
# axs[0].stem(y1)
# axs[0].set_ylabel('y1[n]')
# axs[1].stem(y2)
# axs[1].set_ylabel('y2[n]')
# axs[2].stem(y3)
# axs[2].set_xlabel('n')
# axs[2].set_ylabel('y3[n]')
#
# plt.tight_layout()
# #plt.show()
# fig.savefig("P2_1.eps", format="eps", dpi=300)
#
#
# h2 = [-5, 2, 4, 1, 0, 10, 1, -3, 2]
# y4 = np.convolve(x1, h2, mode='full')
#
# fig, axs = plt.subplots(1, 1, figsize=(12, 3))
# axs.stem(y4)
# axs.set_ylabel('y4[n]')
# axs.set_xlabel('n')
#
# plt.tight_layout()
# #plt.show()
# fig.savefig("P2_2.eps", format="eps", dpi=300)

#
# T = 1.001  # period of response observation
#
# t = np.arange(0, T, 0.001)  # time vector
#
# h_t_5 = np.sin(100 * t) * np.exp(-5 * t)
# h_t_50 = np.sin(100 * t) * np.exp(-50 * t)
#
# input_signal = np.zeros_like(t)
# input_signal[100] = 1  # Impulse at 100ms (index 100 corresponds to 100ms)
#
# y_t_5 = np.convolve(input_signal, h_t_5, mode='full')
# y_t_50 = np.convolve(input_signal, h_t_50, mode='full')
# #output_time = np.arange(0, len(y_t) * 0.001, 0.001)  # Extend time vector for full convolution output
# y_t_5_trimmed = y_t_5[:len(t)]
# y_t_50_trimmed = y_t_50[:len(t)]
#
# fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
# axs[0].plot(t, input_signal)
# axs[0].set_ylabel("Amplitude")
# axs[0].set_title("Impulse Input Signal")
# axs[1].plot(t, h_t_5, label='n=5')
# axs[1].plot(t[:350], np.exp(-5 * t[:350]), color='r', linestyle='dashed')
# axs[1].plot(t[7:65], np.exp(-50 * t[7:65]), color='r', linestyle='dashed')
# axs[1].plot(t, h_t_50, label='n=50')
# axs[1].set_ylabel("Amplitude")
# axs[1].set_title(f'Impulse Response {r"$h(t) = \sin(100t) \cdot e^{-nt}$"}')
# axs[1].legend()
# axs[2].plot(t, y_t_5_trimmed, label='n=5')
# axs[2].plot(t, y_t_50_trimmed, label='n=50')
# axs[2].set_xlabel("Time (s)")
# axs[2].set_ylabel("Amplitude")
# axs[2].set_title("System Output y(t)")
# axs[2].legend()
#
# plt.tight_layout()
# plt.show()
# fig.savefig("P2_3.eps", format="eps", dpi=300)


# ######################################################################################################################
# # Discretization
# ################

# sample_rate, data = wavfile.read("Classic.wav")
#
# # Check the number of bits per sample in the data
# if data.dtype == np.int16:
#     original_bit_depth_classic = 16
# elif data.dtype == np.int32:
#     original_bit_depth_classic = 32
# elif data.dtype == np.float32:
#     original_bit_depth_classic = 32  # Float format typically uses 32 bits
# elif data.dtype == np.float64:
#     original_bit_depth_classic = 64
# else:
#     original_bit_depth_classic = "Unknown"
#
# print(f"Original bit depth: {original_bit_depth_classic} bits")
#
# # Downsample by reducing bit depth to 8-bit and 16-bit
# def downsample_bit_depth(data, Nbits):
#     data_normalized = data / np.max(np.abs(data))
#     # Scale to new range based on desired Nbits
#     if Nbits == 8:
#         data_scaled = (data_normalized * (2**(Nbits - 1) - 1)).astype(np.int8)  # Convert to integer
#     else:
#         data_scaled = (data_normalized * (2 ** (Nbits - 1) - 1)).astype(np.int16)  # Convert to integer
#     # Rescale back to -1 to 1 for audio playback
#     return data_scaled
#
# # Downsample to 8-bit and 16-bit
# data_8bit = downsample_bit_depth(data, 8)
# data_16bit = downsample_bit_depth(data, 16)
#
# # Check the number of bits per sample in the data
# if data_8bit.dtype == np.int16:
#     original_bit_depth_classic = 16
# elif data_8bit.dtype == np.int32:
#     original_bit_depth_classic = 32
# elif data_8bit.dtype == np.float32:
#     original_bit_depth_classic = 32  # Float format typically uses 32 bits
# elif data_8bit.dtype == np.float64:
#     original_bit_depth_classic = 64
# elif data_8bit.dtype == np.int8:
#     original_bit_depth_classic = 8
# else:
#     original_bit_depth_classic = "Unknown"
#
# print(f"Original bit depth: {original_bit_depth_classic} bits")
#
# # print("Playing original audio...")
# # sd.play(data, samplerate=sample_rate)
# # sd.wait()
# #
# # print("Playing 8-bit downsampled audio...")
# # sd.play(data_8bit, samplerate=sample_rate)
# # sd.wait()
#
# filename_acid = 'Acid.wav'
# sample_rate_acid, acid_data = wavfile.read(filename_acid)
#
# # Compare dynamic range
# if acid_data.dtype == np.int16:
#     bit_depth_acid = 16
# elif acid_data.dtype == np.int32:
#     bit_depth_acid = 32
# elif acid_data.dtype == np.float32:
#     bit_depth_acid = 32  # Float format typically uses 32 bits
# elif acid_data.dtype == np.float64:
#     bit_depth_acid = 64
# else:
#     bit_depth_acid = "Unknown"
#
# print(f"Original bit depth: {bit_depth_acid} bits")
#
# # Downsample the audio data to 8-bit and 16-bit
# acid_16bit = downsample_bit_depth(acid_data, 16)
# acid_8bit = downsample_bit_depth(acid_data, 8)
#
# fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
#
# axs[0].plot(np.linspace(0, len(acid_16bit) / sample_rate_acid, num=len(acid_16bit)), acid_16bit, label='Acid.wav')
# axs[0].plot(np.linspace(0, len(data_16bit) / sample_rate, num=len(data_16bit)), data_16bit, label='Classic.wav')
# axs[0].set_ylabel('Amplitude')
# axs[0].set_title('Original Audio')
# axs[0].legend()
# axs[1].plot(np.linspace(0, len(acid_8bit) / sample_rate_acid, num=len(acid_8bit)), acid_8bit, label='Acid.wav')
# axs[1].plot(np.linspace(0, len(data_8bit) / sample_rate, num=len(data_8bit)), data_8bit, label='Classic.wav')
# axs[1].set_xlabel('Time [s]')
# axs[1].set_ylabel('Amplitude')
# axs[1].set_title('Downsampled 8-bit Audio')
# axs[1].legend()
#
# plt.tight_layout()  # Leave space for the main title
# plt.show()
# fig.savefig("P3_1.eps", format="eps", dpi=300)
#
# # print("Playing original audio...")
# # sd.play(acid_data, samplerate=sample_rate_acid)
# # sd.wait()
# #
# # print("Playing 16-bit downsampled audio...")
# # sd.play(acid_16bit, samplerate=sample_rate_acid)
# # sd.wait()
# #
# # print("Playing 8-bit downsampled audio...")
# # sd.play(acid_8bit, samplerate=sample_rate_acid)
# # sd.wait()


# # Parameters
# freq = 10  # frequency of stimuli
# delta_t = 1/1024  # Time step
#
# # Time vector from 0 to 1 - 4*delta_t, with step size delta_t
# t = np.arange(0, 1-delta_t, delta_t)
#
# # Signal
# x = np.sqrt(2) * np.sin(2 * np.pi * freq * t)
# fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
# axs[0].plot(t, x)
# axs[0].set_ylabel("Amplitude")
#
# # Time steps for resampling
# time_steps = [1 / 15, 1 / 20, 1 / 40, 1 / 80, 1 / 160]
#
# axs[1].plot(t, x, label='Original Signal', color='black', linewidth=2)
#
# # Resample and plot each signal
# for step in time_steps:
#     # Create new time vector for each resampling step
#     t_resampled = np.arange(0, 1 - step, step)
#
#     # Generate the resampled signal (interpolation of the original signal)
#     x_resampled = np.sqrt(2) * np.sin(2 * np.pi * freq * t_resampled)
#
#     # Plot the resampled signal
#     axs[1].plot(t_resampled, x_resampled, label=f'Resampled at {1 / step} Hz')
#
# # Adding labels and title
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Amplitude')
# axs[1].legend()
#
# plt.tight_layout()
# plt.show()
# fig.savefig("P3_2.eps", format="eps", dpi=300)
#
#
# sample_rate, y = wavfile.read('Acid.wav')
# print(sample_rate)
#
# # Time vector for the original signal
# t_audio = np.arange(0, len(y) / sample_rate, 1 / sample_rate)
#
# # Resample by taking every 4th point and every 128th point
# y_resampled_4th = y[::4]  # Taking every 4th point
# y_resampled_128th = y[::128]  # Taking every 128th point
#
# # Time vectors for the resampled signals
# t_resampled_4th = t_audio[::4]
# t_resampled_128th = t_audio[::128]
#
# # Plotting the original and resampled signals
# fig, axs = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
#
# # Plot the original signal
# axs.plot(t_audio, y, label='Original Signal', color='black', linewidth=2)
#
# # Plot the resampled signals
# axs.plot(t_resampled_4th, y_resampled_4th, label='Resampled at every 4th point')
# axs.plot(t_resampled_128th, y_resampled_128th, label='Resampled at every 128th point')
#
# # Adding labels and title
# axs.set_xlabel('Time (s)')
# axs.set_ylabel('Amplitude')
# axs.legend()
#
# plt.tight_layout()
# plt.show()
# fig.savefig("P3_3_2.eps", format="eps", dpi=300)


# ######################################################################################################################
# # Correlation
# ################
#
# Read the audio file
Fs, audioSignal = wavfile.read('Balloon_anechoic.wav')

# Normalize the audio signal (optional, depending on your data format)
audioSignal = audioSignal[:, 0] / np.max(np.abs(audioSignal[:, 0]))

# Cross-correlate the audio signal with itself
correlation = correlate(audioSignal, audioSignal, mode='full')
correlation /= np.max(np.abs(correlation))  # Normalize to scale between -1 and 1

# Define the time axis in seconds for the correlation plot
# Calculate the lag indices (in samples)
lags = np.arange(-len(audioSignal) + 1, len(audioSignal))

# Convert lag indices to time by dividing by the sampling frequency
timeLags = lags / Fs

# Plot the cross-correlation with the x-axis in time (seconds)
fig, axs = plt.subplots(1, 1, figsize=(12, 3))

axs.plot(timeLags, correlation)
axs.set_xlabel('Time Lag (seconds)')
axs.set_ylabel('Normalized Cross-Correlation')

plt.tight_layout()  # Leave space for the main title
plt.show()
fig.savefig("P4_1.eps", format="eps", dpi=300)


# Constants
c = 342  # Speed of sound in air in m/s

# Read the echo audio file
Fs, audioSignal_echo = wavfile.read('Balloon_echo.wav')

# Normalize the audio signal
audioSignal_echo = audioSignal_echo[:, 0] / np.max(np.abs(audioSignal_echo[:, 0]))

# Perform autocorrelation to find the echo delay
correlation = correlate(audioSignal_echo, audioSignal_echo, mode='full')
correlation /= np.max(np.abs(correlation))  # Normalize correlation for easy peak detection

# Define the lag indices
lags = np.arange(-len(audioSignal_echo) + 1, len(audioSignal_echo))
timeLags = lags / Fs  # Convert lag indices to time (seconds)

# Find the time delay by locating the first significant peak after zero lag
# We take the positive side of timeLags for analysis
half_len = len(correlation) // 2
time_after_zero = timeLags[half_len:]
correlation_after_zero = correlation[half_len:]

# Plot the cross-correlation with the x-axis in time (seconds)
fig, axs = plt.subplots(1, 1, figsize=(12, 3))

axs.plot(timeLags, correlation)
axs.set_xlabel('Time Lag (seconds)')
axs.set_ylabel('Normalized Cross-Correlation')

plt.tight_layout()  # Leave space for the main title
plt.show()
fig.savefig("P4_2.eps", format="eps", dpi=300)

td = 0.5

# Calculate the distance to the wall
d = c * td / 2

# Print results
print(f"Estimated Echo Delay, td: {td:.3f} seconds")
print(f"Estimated Distance to Wall, d: {d:.2f} meters")


# Read the echo audio file
Fs, audioSignal = wavfile.read('Balloon_echo_or_not.wav')

sd.play(audioSignal, samplerate=Fs)
sd.wait()

# Normalize the audio signal
audioSignal = audioSignal[:, 0] / np.max(np.abs(audioSignal[:, 0]))

# Perform autocorrelation to find the echo delay
correlation = correlate(audioSignal, audioSignal, mode='full')
correlation /= np.max(np.abs(correlation))  # Normalize correlation for easy peak detection

# Define the lag indices
lags = np.arange(-len(audioSignal) + 1, len(audioSignal))
timeLags = lags / Fs  # Convert lag indices to time (seconds)

# Plot the cross-correlation with the x-axis in time (seconds)
fig, axs = plt.subplots(1, 1, figsize=(12, 3))

axs.plot(timeLags, correlation)
axs.set_xlabel('Time Lag (seconds)')
axs.set_ylabel('Normalized Cross-Correlation')

plt.tight_layout()  # Leave space for the main title
plt.show()
fig.savefig("P4_3.eps", format="eps", dpi=300)

