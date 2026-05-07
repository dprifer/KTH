import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from cycler import cycler


def set_publication_style():
    tol_colors = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
    ]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        'axes.prop_cycle': cycler(color=tol_colors)
    })


def plot_track_data(S, Y, Z):
    set_publication_style()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6.5))

    # --- Lateral ---
    axs[0].plot(S, Y)
    axs[0].set_title("Lateral track defect", pad=6)
    axs[0].set_ylabel("Y [mm]")
    axs[0].grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[0].minorticks_on()
    axs[0].grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.2)
    axs[0].set_xlim(S[0], S[-1])

    # --- Vertical ---
    axs[1].plot(S, Z)
    axs[1].set_title("Vertical track defect", pad=6)
    axs[1].set_xlabel("S [m]")
    axs[1].set_ylabel("Z [mm]")
    axs[1].grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1].minorticks_on()
    axs[1].grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.2)
    axs[1].set_xlim(S[0], S[-1])

    # Cleaner layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, axs


def plot_spectral_and_filtering(
    f, Px_Y, Px_Z, Px_YF, Px_ZF,
    S, Y, Z, YF, ZF, Nstripes, df, wavelength_class
):
    set_publication_style()

    if wavelength_class == 'D1':
        Lmin = 3
        Lmax = 25
    elif wavelength_class == 'D2':
        Lmin = 25
        Lmax = 70

    L = 1 / f

    # -----------------------------
    # ERRI DSP formulation
    # -----------------------------
    Fc = 0.8246  # rad/m
    Fr = 0.0206  # rad/m

    Ay_low  = 2.119e-7
    Ay_high = 6.125e-7
    Az_low  = 4.032e-7
    Az_high = 1.08e-6

    Num = Fc**2
    Den = ((2*np.pi*f)**2 + Fr**2) * ((2*np.pi*f)**2 + Fc**2)

    ERRI_low_Y  = Ay_low  * Num / Den * 1e6 * (2*np.pi)
    ERRI_high_Y = Ay_high * Num / Den * 1e6 * (2*np.pi)
    ERRI_low_Z  = Az_low  * Num / Den * 1e6 * (2*np.pi)
    ERRI_high_Z = Az_high * Num / Den * 1e6 * (2*np.pi)

    # Colors (muted, publication-friendly)
    ref_color   = "#7f8c8d"  # soft gray
    filt_color  = "#c0392b"  # muted red

    # =============================
    # FIGURE 2: Spectral Analysis
    # =============================
    fig2, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    fig2.suptitle(f"Spectral Analysis", fontsize=13)

    xlabel = f"L [m] (npts = {2*Nstripes}; df = {df:.3g} 1/m)"

    # --- Lateral ---
    axs[0].loglog(L, ERRI_low_Y,  color=ref_color, linestyle="--", label="ERRI low")
    axs[0].loglog(L, ERRI_high_Y, color=ref_color, linestyle="-.", label="ERRI high")
    axs[0].loglog(L, Px_Y, label="Track")

    axs[0].set_title("Lateral track defects – PSD")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("DSP Y [mm²/(1/m)]")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(loc="lower right")

    # --- Vertical ---
    axs[1].loglog(L, ERRI_low_Z,  color=ref_color, linestyle="--", label="ERRI low")
    axs[1].loglog(L, ERRI_high_Z, color=ref_color, linestyle="-.", label="ERRI high")
    axs[1].loglog(L, Px_Z, label="Track")

    axs[1].set_title("Vertical track defects – PSD")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("DSP Z [mm²/(1/m)]")
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend(loc="lower right")

    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # =============================
    # FIGURE 3: Filtering check
    # =============================
    fig3, axs = plt.subplots(2, 2, figsize=(12, 8))

    fig3.suptitle(f"Filtering in domain {Lmin}–{Lmax} m", fontsize=13)

    # --- Lateral time domain ---
    axs[0, 0].plot(S, Y, label="Raw")
    axs[0, 0].plot(S, YF, color=filt_color,  label=f"{Lmin}–{Lmax} m")

    axs[0, 0].set_title("Lateral track defect")
    axs[0, 0].set_xlabel("S [m]")
    axs[0, 0].set_ylabel("Y [mm]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].set_xlim(S[0], S[-1])

    # --- Lateral PSD ---
    axs[0, 1].loglog(L, Px_Y, label="Raw")
    axs[0, 1].loglog(L, Px_YF, color=filt_color,  label=f"{Lmin}–{Lmax} m")

    axs[0, 1].set_title("Lateral track defects – PSD")
    axs[0, 1].set_xlabel(xlabel)
    axs[0, 1].set_ylabel("DSP Y [mm²/(1/m)]")
    axs[0, 1].grid(True, which="both", alpha=0.3)
    axs[0, 1].legend(loc="lower right")

    # --- Vertical time domain ---
    axs[1, 0].plot(S, Z, label="Raw")
    axs[1, 0].plot(S, ZF, color=filt_color,  label=f"{Lmin}–{Lmax} m")

    axs[1, 0].set_title("Vertical track defect")
    axs[1, 0].set_xlabel("S [m]")
    axs[1, 0].set_ylabel("Z [mm]")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(loc="lower right")
    axs[1, 0].set_xlim(S[0], S[-1])

    # --- Vertical PSD ---
    axs[1, 1].loglog(L, Px_Z, label="Raw")
    axs[1, 1].loglog(L, Px_ZF, color=filt_color,  label=f"{Lmin}–{Lmax} m")

    axs[1, 1].set_title("Vertical track defects – PSD")
    axs[1, 1].set_xlabel(xlabel)
    axs[1, 1].set_ylabel("DSP Z [mm²/(1/m)]")
    axs[1, 1].grid(True, which="both", alpha=0.3)
    axs[1, 1].legend(loc="lower right")

    fig3.tight_layout(rect=[0, 0, 1, 0.95])

    # =============================
    # Saving
    # =============================
    # folder = os.path.dirname(filepath)
    # os.makedirs(folder, exist_ok=True)
    # base = os.path.splitext(os.path.basename(filepath))[0]
    #
    # fig2.savefig(os.path.join(folder, f"{base}_DSP.pdf"), bbox_inches="tight")
    # fig3.savefig(os.path.join(folder, f"{base}_CheckFiltering.pdf"), bbox_inches="tight")

    return fig2, fig3


def plot_UIC518_results(S, S_sec, Y, Z, Etyp_Y, Ymax_d1, Etyp_Z, Zmax_d1, Lsec, Q1_lim_Y_SD, Q1_lim_Z_SD, Q2_lim_Y_SD, Q2_lim_Z_SD, Q1_lim_Y_max, Q1_lim_Z_max, Q2_lim_Y_max, Q2_lim_Z_max, Q3_lim_Y_max, Q3_lim_Z_max, Vmax):

    # colors
    marker_color = color="#2c3e50"
    QN1_color = "green"
    QN2_color = "orange"
    QN3_color = "yellow"

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Track quality (UIC 518 Appendix D) at Vmax = {Vmax} km/h in D1 (3–25 m) domain", fontsize=14)

    ax = axes[0, 0]
    ax.plot(S, Y, color="grey", alpha=0.4, label="3–25 m band-pass")
    ax.plot(S_sec, Ymax_d1, "*-", ms=4, color=marker_color, label="Max / section")
    ax.plot(S_sec, -Ymax_d1, "*-", ms=4, color=marker_color)
    ax.axhline(Q3_lim_Y_max, color=QN3_color, linestyle="-", label="QN3 limit")
    ax.axhline(-Q3_lim_Y_max, color=QN3_color, linestyle="-")
    ax.axhline(Q2_lim_Y_max, color=QN2_color, linestyle="-", label="QN2 limit")
    ax.axhline(-Q2_lim_Y_max, color=QN2_color, linestyle="-")
    ax.axhline(Q1_lim_Y_max, color=QN1_color, linestyle="-", label="QN1 limit")
    ax.axhline(-Q1_lim_Y_max, color=QN1_color, linestyle="-")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Lateral defect peak values")
    ax.set_xlim(S[0], S[-1])
    # ax.legend(loc="lower right")

    ax = axes[1, 0]
    ax.scatter((np.arange(len(Etyp_Y)) + 0.5) * Lsec, Etyp_Y, s=20, color=marker_color, edgecolor="k", linewidth=0.3)
    ax.axhspan(0, Q1_lim_Y_SD, color=QN1_color, alpha=0.15)
    ax.axhspan(Q1_lim_Y_SD, Q2_lim_Y_SD, color=QN2_color, alpha=0.15)
    ax.axhspan(Q2_lim_Y_SD, max(Etyp_Y) * 1.2, color=QN3_color, alpha=0.15)
    ax.set_ylabel("SD Y [mm]")
    ax.set_title("Lateral SD per section")
    ax.set_xlim(S[0], S[-1])
    ax.set_ylim(0, max(Etyp_Y) * 1.2)

    ax = axes[0, 1]
    ax.plot(S, Z, color="grey", alpha=0.4, label="3–25 m band-pass")
    ax.plot(S_sec, Zmax_d1, "*-", ms=4, color=marker_color, label="Max / section")
    ax.plot(S_sec, -Zmax_d1, "*-", ms=4, color=marker_color)
    ax.axhline(Q3_lim_Z_max, color=QN3_color, linestyle="-", label="QN3 limit")
    ax.axhline(-Q3_lim_Z_max, color=QN3_color, linestyle="-")
    ax.axhline(Q2_lim_Z_max, color=QN2_color, linestyle="-", label="QN2 limit")
    ax.axhline(-Q2_lim_Z_max, color=QN2_color, linestyle="-")
    ax.axhline(Q1_lim_Z_max, color=QN1_color, linestyle="-", label="QN1 limit")
    ax.axhline(-Q1_lim_Z_max, color=QN1_color, linestyle="-")
    ax.set_ylabel("Z [mm]")
    ax.set_title("Vertical defect peak values")
    ax.set_xlim(S[0], S[-1])
    # ax.legend(loc="lower right")

    ax = axes[1, 1]
    ax.scatter((np.arange(len(Etyp_Z)) + 0.5) * Lsec, Etyp_Z, s=20, color=marker_color, edgecolor="k", linewidth=0.3)
    ax.axhspan(0, Q1_lim_Z_SD, color=QN1_color, alpha=0.15)
    ax.axhspan(Q1_lim_Z_SD, Q2_lim_Z_SD, color=QN2_color, alpha=0.15)
    ax.axhspan(Q2_lim_Z_SD, max(Etyp_Z) * 1.2, color=QN3_color, alpha=0.15)
    ax.set_ylabel("SD Z [mm]")
    ax.set_title("Vertical SD per section")
    ax.set_xlim(S[0], S[-1])
    ax.set_ylim(0, max(Etyp_Z) * 1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_UIC518_classification(CL80_QNY, CL120_QNY, CL160_QNY, CL200_QNY, CL300_QNY,
                          CL80_QNZ, CL120_QNZ, CL160_QNZ, CL200_QNZ, CL300_QNZ,
                          S_sec):

    set_publication_style()

    labels = ["QN1", "QN2", "QN3", "EXCL"]
    class_colors = {
        1: "green",  # QN1 (good)
        2: "orange",  # QN2
        3: "yellow",  # QN3
        4: "red",  # EXCL (bad)
    }

    # Left column (Y-direction)
    CL_all_Y = [
        ("80 km/h", CL80_QNY),
        ("120 km/h", CL120_QNY),
        ("160 km/h", CL160_QNY),
        ("200 km/h", CL200_QNY),
        ("300 km/h", CL300_QNY),
    ]

    # Right column (Z-direction)
    CL_all_Z = [
        ("80 km/h", CL80_QNZ),
        ("120 km/h", CL120_QNZ),
        ("160 km/h", CL160_QNZ),
        ("200 km/h", CL200_QNZ),
        ("300 km/h", CL300_QNZ),
    ]

    # Create 2-column layout
    fig, axes = plt.subplots(len(CL_all_Y), 2, figsize=(16, 10), sharex=True, sharey=True)

    for i, ((speed_label, CL_Y), (_, CL_Z)) in enumerate(zip(CL_all_Y, CL_all_Z)):
        colors_Y = [class_colors[c] for c in CL_Y]
        colors_Z = [class_colors[c] for c in CL_Z]
        # --- Left column (Y)
        ax = axes[i, 0]
        ax.scatter(
            S_sec, CL_Y,
            c=colors_Y,
            s=35, edgecolor='k', linewidth=0.3
        )
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(labels)
        ax.set_ylabel(speed_label)
        ax.set_ylim(0.5, 4.5)
        ax.grid(True, alpha=0.3)

        # --- Right column (Z)
        ax = axes[i, 1]
        ax.scatter(
            S_sec, CL_Z,
            c=colors_Z,
            s=35, edgecolor='k', linewidth=0.3
        )
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(labels)
        ax.set_ylim(0.5, 4.5)
        ax.grid(True, alpha=0.3)

    # Column titles
    axes[0, 0].set_title("Y direction")
    axes[0, 1].set_title("Z direction")

    # Shared x-label
    axes[-1, 0].set_xlabel("Track position S [m]")
    axes[-1, 1].set_xlabel("Track position S [m]")

    fig.suptitle("Track Quality Classification Along Route", fontsize=14)
    plt.tight_layout()

    return fig








