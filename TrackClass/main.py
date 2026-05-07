from TrackClass.analysis import *
from TrackClass.plots import *
import os


filepath = r"C:\Users\prife\OneDrive - KTH\KTH\04 Research\02 Publications\Potential\TrackQuality\Track Analysis\UIC518\4700_Stuttgart - Ulm_To_0_V1_R119031405_StrNr_4700-1_Rev0.tre"
outputDir = r"C:\Users\prife\OneDrive - KTH\KTH\04 Research\02 Publications\Potential\TrackQuality\Track Analysis\UIC518\output"
os.makedirs(outputDir, exist_ok=True)

e = load_tre_file(filepath)

S = e[:, 0]  # in m
Y = e[:, 1]  # in mm
Z = e[:, 2]  # in mm

fig, axs = plot_track_data(S, Y, Z)
# fig.savefig(os.path.join(outputDir, "track_data.pdf"), bbox_inches="tight")

fig2, fig3, fig4, fig5 = UIC518_classification(S, Y, Z, 100, 200)

# fig2.savefig(os.path.join(outputDir, f"DSP.pdf"), bbox_inches="tight")
# fig3.savefig(os.path.join(outputDir, f"CheckFiltering.pdf"), bbox_inches="tight")
plt.show()


