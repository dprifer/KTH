import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, rfftfreq, fftfreq
# from scipy.signal.windows import hann
from scipy.signal import csd, welch, windows
from scipy.io import loadmat
#%%
# Load data from ImpTube12.mat
data = loadmat('ImpTube12.mat')
Mic12 = data['Mic12']
fs = int(data['fs'][0][0])

signal1 = Mic12[:, 0]
signal2 = Mic12[:, 1]

# Define Welch parameters
overlap_factor = 0.5  # 50% overlap
#%%
# Calculate and plot PSDs for each signal
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
i = 0
nfft = 7680

window = np.hanning(nfft)  # Hanning window
overlap = int(nfft * overlap_factor)  # 50% overlap

# Compute the Welch PSD
f, psd1 = welch(signal1, fs=fs, window=window, nperseg=nfft, noverlap=overlap, scaling='density')
_, psd2 = welch(signal2, fs=fs, window=window, nperseg=nfft, noverlap=overlap, scaling='density')
f12, S12 = csd(signal1, signal2, fs=fs, window=window, nperseg=nfft, noverlap=overlap)

C_12 = np.abs(S12) ** 2 / (psd1 * psd2)

# Plot PSD
axs[0].semilogy(f, psd1, label=f'Signal 1')
axs[0].semilogy(f, psd2, label=f'Signal 2')
axs[0].set_title(f"PSD (NFFT = {nfft})")
axs[0].set_ylabel("Power/Freq. (dB/Hz)")
axs[0].legend()
axs[0].grid()


axs[1].plot(f, C_12)
axs[1].set_title(f"Coherence (NFFT = {nfft})")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Coherence")
axs[1].grid()

plt.tight_layout()
plt.savefig(f'E4_P1_1.eps', format='eps')  # Save the figure in EPS format
plt.show()