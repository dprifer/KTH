import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from scipy.signal.windows import hann
import os
from TrackClass.plots import *


def load_tre_file(filepath, skip_header=20):
    with open(filepath, 'r') as f:
        for _ in range(skip_header):
            next(f)
        data = np.loadtxt(f)
    return data


def percentile(signal, percent):
    n_percent = int(np.ceil(percent / 100 * len(signal)))
    return np.sort(signal)[n_percent - 1]



def compute_psd(signal, fs, noverlap, nfft):

    window = 'hamming'  # this is a good default for track analysis (better peak detection and amplitude accuracy)

    f, Pxx = welch(
        signal,
        fs=fs,
        window=window,
        nperseg=nfft,  # == len(window)
        noverlap=noverlap,
        nfft=nfft,  # must equal nperseg to avoid zero-padding
        scaling="density",
        detrend=False,
        return_onesided=True
    )

    return f, Pxx


def filtering(S, Y, Z, wavelength_class):

    # ----------------------------
    # Sampling properties
    # ----------------------------
    N = len(S)
    dS = np.mean(np.diff(S))
    fS = 1 / dS

    # ----------------------------
    # Filtering (3–25 m / D1 domain according to EN13848)
    # ----------------------------

    if wavelength_class == 'D1':

        Lmin = 3
        Lmax = 25

        Lm = np.array([Lmax, Lmin])
        Wm = 1.0 / Lm  # spatial frequencies

        # Design Butterworth bandpass filter
        b, a = butter(N=2, Wn=Wm, btype='bandpass', fs=fS)  # for bandpass, N=2 --> 4-pole

        # Zero-phase filtering
        YF = filtfilt(b, a, Y)
        ZF = filtfilt(b, a, Z)

        return N, dS, fS, YF, ZF

    elif wavelength_class == 'D2':
        Lmin = 25
        Lmax = 70

        Lm = np.array([Lmax, Lmin])
        Wm = 1.0 / Lm  # spatial frequencies

        # Design Butterworth bandpass filter
        b, a = butter(N=2, Wn=Wm, btype='bandpass', fs=fS)  # for bandpass, N=2 --> 4-pole

        # Zero-phase filtering
        YF = filtfilt(b, a, Y)
        ZF = filtfilt(b, a, Z)

        return N, dS, fS, YF, ZF

    else:
        raise ValueError('INPUT "type" must be "D1" or "D2"')


def spectralAnalysis(S, fS, Y, YF, Z, ZF, wavelength_class):
    nFFT = 512
    noverlap = int(nFFT / 4)  # 128, i.e. 25% overlap

    df = fS / nFFT  # frequency resolution

    Nstripes = int(nFFT / 2)  # number of frequency bins

    f, Px_Y = compute_psd(Y, fS, noverlap, nFFT)
    _, Px_YF = compute_psd(YF, fS, noverlap, nFFT)
    _, Px_Z = compute_psd(Z, fS, noverlap, nFFT)
    _, Px_ZF = compute_psd(ZF, fS, noverlap, nFFT)

    fig2, fig3 = plot_spectral_and_filtering(
        f, Px_Y, Px_Z, Px_YF, Px_ZF,
        S, Y, Z, YF, ZF, Nstripes, df, wavelength_class)

    return fig2, fig3


def en14363_limits(Vmax):
    """
    Return TL90 (target level for 90 %-value of standard deviation) and QN3 (quality limit for discrete track defects) limits from EN14363 Annex M table M.3
    """
    Speed_range = np.array([120, 160, 200, 230, 300], dtype=float)  # differs from EN13848 speed classes
    EN14363_AnnexM = np.array([
        [1.80, 1.40, 1.15, 1.05, 0.85],  # DefZ Min TL90
        [2.50, 1.85, 1.60, 1.45, 1.10],  # DefZ Max TL90
        [1.05, 0.75, 0.70, 0.65, 0.50],  # DefY Min TL90
        [1.45, 1.00, 0.90, 0.80, 0.65],  # DefY Max TL90
        [16.0, 13.0, 12.0, 10.0, 10.0],  # DefZ Max QN3
        [13.0, 10.0, 9.0, 8.0, 8.0],     # DefY Max QN3
    ])
    speed_class = 1 + np.sum(Speed_range < Vmax)
    speed_class = int(max(1, min(speed_class, Speed_range.size)))

    TL90_limitZ_lower = EN14363_AnnexM[0, speed_class - 1]
    TL90_limitZ_upper = EN14363_AnnexM[1, speed_class - 1]
    TL90_limitY_lower = EN14363_AnnexM[2, speed_class - 1]
    TL90_limitY_upper = EN14363_AnnexM[3, speed_class - 1]
    QN3_limitZ = EN14363_AnnexM[4, speed_class - 1]
    QN3_limitY = EN14363_AnnexM[5, speed_class - 1]

    return (TL90_limitZ_lower, TL90_limitZ_upper,
            TL90_limitY_lower, TL90_limitY_upper,
            QN3_limitZ, QN3_limitY)


def UIC518_classification(S, Y, Z, Lsec, Vmax):
    N, dS, fS, YF_d1, ZF_d1 = filtering(S, Y, Z, wavelength_class='D1')
    _, _, _, YF_d2, ZF_d2 = filtering(S, Y, Z, wavelength_class='D2')
    print(f"number of samples = {N}, sampling step = {dS:.4f}, spatial sampling frequency = {fS:.4f}")

    # Plot fintering effect for checking and comparison with ERRI low & high
    fig2, fig3 = spectralAnalysis(S, fS, Y, YF_d1, Z, ZF_d1, wavelength_class='D1')
    # _, fig4 = spectralAnalysis(S, fS, Y, YF_d2, Z, ZF_d2, wavelength_class='D2')

    N = int(np.floor(Lsec / dS))  # samples per segment
    m = int(np.floor(len(Y) / N))  # number of full segments

    Xz = np.arange(1, m + 1) * Lsec
    S_sec = Xz - np.floor(Lsec / 2.0)

    # Reshape into segments ==> N rows and m columns
    YFz_d1 = YF_d1[:N * m].reshape(m, N).T
    YFz_d2 = YF_d2[:N * m].reshape(m, N).T

    ZFz_d1 = ZF_d1[:N * m].reshape(m, N).T
    ZFz_d2 = ZF_d2[:N * m].reshape(m, N).T

    # -----------------------------
    # Statistics
    # -----------------------------
    Etyp_Y = np.std(YFz_d1, axis=0)  # Standard deviation of filtered lateral track irregularities per section
    Etyp_Z = np.std(ZFz_d1, axis=0)  # Standard deviation of filtered vertical track irregularities per section

    Ymax_d1 = np.max(np.abs(YFz_d1), axis=0)  # peak values per segment
    Zmax_d1 = np.max(np.abs(ZFz_d1), axis=0)

    Ymax_d2 = np.max(np.abs(YFz_d2), axis=0)
    Zmax_d2 = np.max(np.abs(ZFz_d2), axis=0)

    # -----------------------------
    # Criteria values
    # -----------------------------
    VV = np.array([80, 120, 160, 200, 300])

    Q1ET_Yc = np.array([1.5, 1.2, 1, 0.8, 0.7])
    Q1CR_Yc = np.array([12, 8, 6, 5, 4])

    Q1ET_Zc = np.array([2.3, 1.8, 1.4, 1.2, 1])
    Q1CR_Zc = np.array([12, 8, 6, 5, 4])

    Q2ET_Yc = np.array([1.8, 1.5, 1.3, 1.1, 1])
    Q2CR_Yc = np.array([14, 10, 8, 7, 6])

    Q2ET_Zc = np.array([2.6, 2.1, 1.7, 1.5, 1.3])
    Q2CR_Zc = np.array([16, 12, 10, 9, 8])

    Q3CR_Yc = 1.3 * Q2CR_Yc
    Q3CR_Zc = 1.3 * Q2CR_Zc

    idx = np.searchsorted(VV, Vmax, side='left')

    Q1_lim_Y_SD = Q1ET_Yc[idx]
    Q1_lim_Z_SD = Q1ET_Zc[idx]
    Q2_lim_Y_SD = Q2ET_Yc[idx]
    Q2_lim_Z_SD = Q2ET_Zc[idx]

    Q1_lim_Y_max = Q1CR_Yc[idx]
    Q1_lim_Z_max = Q1CR_Zc[idx]
    Q2_lim_Y_max = Q2CR_Yc[idx]
    Q2_lim_Z_max = Q2CR_Zc[idx]
    Q3_lim_Y_max = Q3CR_Yc[idx]
    Q3_lim_Z_max = Q3CR_Zc[idx]

    # Criterion on newly adjusted track, long wavelength in Sweden (Table K11.5 & K11.7)
    ACR_Y = np.array([5])
    ACR_Z = np.array([7])

    # Before planned maintenance criterion on long wavelength Sweden (Table K11.5 & K11.7)
    BCR_Y = np.array([9])
    BCR_Z = np.array([12])

    Nzone = len(Xz)

    def classify(Etyp, maxc, Q1, Q2, Q3):
        """
        Returns class per segment:
        1 = QN1
        2 = QN2
        3 = QN3
        4 = EXCLUDED
        """
        CL = np.zeros(len(Etyp), dtype=int)
        CL[maxc >= Q3] = 4
        mask_qn3 = (maxc < Q3) & (Etyp >= Q2)
        CL[mask_qn3] = 3
        mask_qn2 = (maxc < Q3) & (Etyp >= Q1) & (Etyp < Q2)
        CL[mask_qn2] = 2
        mask_qn1 = (maxc < Q3) & (Etyp < Q1)
        CL[mask_qn1] = 1

        return CL

    # -----------------------------
    # Lateral classification (Y)
    # -----------------------------
    CL80_QNY = classify(Etyp_Y, Ymax_d1, Q1ET_Yc[0], Q2ET_Yc[0], Q3CR_Yc[0])
    CL120_QNY = classify(Etyp_Y, Ymax_d1, Q1ET_Yc[1], Q2ET_Yc[1], Q3CR_Yc[1])
    CL160_QNY = classify(Etyp_Y, Ymax_d1, Q1ET_Yc[2], Q2ET_Yc[2], Q3CR_Yc[2])
    CL200_QNY = classify(Etyp_Y, Ymax_d1, Q1ET_Yc[3], Q2ET_Yc[3], Q3CR_Yc[3])
    CL300_QNY = classify(Etyp_Y, Ymax_d1, Q1ET_Yc[4], Q2ET_Yc[4], Q3CR_Yc[4])

    # -----------------------------
    # Vertical classification (Z)
    # -----------------------------
    CL80_QNZ = classify(Etyp_Z, Zmax_d1, Q1ET_Zc[0], Q2ET_Zc[0], Q3CR_Zc[0])
    CL120_QNZ = classify(Etyp_Z, Zmax_d1, Q1ET_Zc[1], Q2ET_Zc[1], Q3CR_Zc[1])
    CL160_QNZ = classify(Etyp_Z, Zmax_d1, Q1ET_Zc[2], Q2ET_Zc[2], Q3CR_Zc[2])
    CL200_QNZ = classify(Etyp_Z, Zmax_d1, Q1ET_Zc[3], Q2ET_Zc[3], Q3CR_Zc[3])
    CL300_QNZ = classify(Etyp_Z, Zmax_d1, Q1ET_Zc[4], Q2ET_Zc[4], Q3CR_Zc[4])

    fig4 = plot_UIC518_results(S, S_sec, Y, Z, Etyp_Y, Ymax_d1, Etyp_Z, Zmax_d1, Lsec, Q1_lim_Y_SD, Q1_lim_Z_SD, Q2_lim_Y_SD, Q2_lim_Z_SD, Q1_lim_Y_max, Q1_lim_Z_max, Q2_lim_Y_max, Q2_lim_Z_max, Q3_lim_Y_max, Q3_lim_Z_max, Vmax)

    fig5 = plot_UIC518_classification(CL80_QNY, CL120_QNY, CL160_QNY, CL200_QNY, CL300_QNY,
                          CL80_QNZ, CL120_QNZ, CL160_QNZ, CL200_QNZ, CL300_QNZ,
                          S_sec)


    return fig2, fig3, fig4, fig5



