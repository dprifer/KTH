import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, wavfile
from scipy.stats import skew, kurtosis
import os


current_dir = os.getcwd()


#######################################################################################################################
## Part 1_1
############

# # Audio file names
# file_names = ['Sound1.wav', 'Sound2.wav', 'Sound3.wav', 'Sound4.wav']

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


#######################################################################################################################
## Part 1_2
############

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


#######################################################################################################################
## Part 1_3
############

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



#######################################################################################################################
## Convolution
#################

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
# y5 = np.convolve(x1+x2, h, mode='full')
# y6 = y1 + y2
#
# fig, axs = plt.subplots(2, 1, figsize=(12, 3))
# axs[0].stem(y5)
# axs[1].stem(y6)
# axs[0].set_ylabel('y[n]')
# axs[1].set_ylabel('y[n]')
# axs[0].set_xlabel('n')
# axs[1].set_xlabel('n')
#
# plt.tight_layout()
# plt.show()


T = 1.001  # period of response observation
n = 5

t = np.arange(0, T, 0.001)  # time vector

h_t = np.sin(100 * t) * np.exp(-n * t)

input_signal = np.zeros_like(t)
input_signal[100] = 1  # Impulse at 100ms (index 100 corresponds to 100ms)

y_t = np.convolve(input_signal, h_t, mode='full') * (t[1] - t[0])  # Scaling by dt
#output_time = np.arange(0, len(y_t) * 0.001, 0.001)  # Extend time vector for full convolution output
y_t_trimmed = y_t[:len(t)]

plt.figure(figsize=(12, 8))

# Input plot
plt.subplot(3, 1, 1)
plt.plot(t, input_signal)
plt.title("Impulse Input Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Impulse Response plot
plt.subplot(3, 1, 2)
plt.plot(t, h_t)
plt.title("Impulse Response h(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Output plot
plt.subplot(3, 1, 3)
plt.plot(t, y_t_trimmed)
plt.title("System Output y(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()


