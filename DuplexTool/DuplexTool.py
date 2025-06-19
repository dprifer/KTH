from openpyxl import Workbook
import pandas as pd
import streamlit as st


class ParameterRange:
    def __init__(self, name, values=None, start=None, stop=None, step=None):
        self.name = name
        if values is not None:  # Explicit list mode
            self.values = list(dict.fromkeys(values))
        elif None not in (start, stop, step):  # Range mode
            self.values = self._generate_range(start, stop, step)
        else:
            raise ValueError("Must provide either values list or start/stop/step")

    def _generate_range(self, start, stop, step):
        values = []
        current = start
        while current <= stop + 1e-9:  # Account for floating-point precision
            values.append(round(current, 9))
            current += step
        return list(dict.fromkeys(values))  # Remove potential duplicates

    def get_values(self):
        for value in self.values:
            yield value

    def size(self):
        return len(self.values)


class SpringFormulas:
    @staticmethod
    def R(G, d, D, n):
        return (G * d**4) / (8 * D**3 * n)

    @staticmethod
    def Lc(d, n):
        tol = 0.25 if d < 32 else 0.3 if d < 42 else 0.4
        return (n+1.5-0.3)*(d+tol)

    @staticmethod
    def tauc(L0, Lc, G, d, D, n):
        sc = L0-Lc
        return (G * d * sc) / (3.1415 * n * D**2)

    @staticmethod
    def w(D, d):
        return D / d

    @staticmethod
    def Sn(L0, Lc, n, D, d):
        Sa = 0.04 * n * (d + D)
        return L0-(Sa+Lc)

    @staticmethod
    def L2(L0, F2, R1, R2):
        return L0-F2/(R1+R2)

    @staticmethod
    def s2(L0, L2):
        return L0-L2


def find_springs(F2, max_OD, max_height, max_stress, min_Sn_s2_gap, min_w, max_w, stiffness_range, param_ranges, G):
    total_iterations = (
            param_ranges['L0'].size() *
            param_ranges['d1'].size() *
            param_ranges['D1'].size() *
            param_ranges['n1'].size() *
            param_ranges['d2'].size() *
            param_ranges['D2'].size() *
            param_ranges['n2'].size()
    )

    iteration = 0
    good_springs = []
    counter = 0

    for L0 in param_ranges['L0'].get_values():
        print(f'Spring: L0={L0}')
        for d1 in param_ranges['d1'].get_values():
            for D1 in param_ranges['D1'].get_values():
                if D1 + d1 > max_OD:
                    iteration += 1
                    continue
                for n1 in param_ranges['n1'].get_values():
                    for d2 in param_ranges['d2'].get_values():
                        for D2 in param_ranges['D2'].get_values():
                            for n2 in param_ranges['n2'].get_values():
                                iteration += 1
                                if D2 + d2 > D1 - d1 - 12:
                                    continue

                                R1 = SpringFormulas.R(G, d1, D1, n1)
                                R2 = SpringFormulas.R(G, d2, D2, n2)
                                if not (stiffness_range[0] <= R1 + R2 <= stiffness_range[1]):
                                    continue

                                Lc1 = SpringFormulas.Lc(d1, n1)
                                tauc1 = SpringFormulas.tauc(L0, Lc1, G, d1, D1, n1)
                                w1 = SpringFormulas.w(D1, d1)
                                Sn1 = SpringFormulas.Sn(L0, Lc1, n1, D1, d1)

                                Lc2 = SpringFormulas.Lc(d2, n2)
                                tauc2 = SpringFormulas.tauc(L0, Lc2, G, d2, D2, n2)
                                w2 = SpringFormulas.w(D2, d2)
                                Sn2 = SpringFormulas.Sn(L0, Lc2, n2, D2, d2)
                                L2 = SpringFormulas.L2(L0, F2, R1, R2)
                                s2 = SpringFormulas.s2(L0, L2)

                                if not (
                                        Sn1 - min_Sn_s2_gap > s2 and
                                        Sn2 - min_Sn_s2_gap > s2 and
                                        L2 < max_height and
                                        min_w < w1 < max_w and
                                        min_w < w2 < max_w and
                                        tauc1 < max_stress and
                                        tauc2 < max_stress
                                ):
                                    continue

                                spring = [L0, Lc1, Lc2, d1, d2, D1, D2, n1, n2, R1, R2,
                                          tauc1, tauc2, w1, w2, Sn1, Sn2, L2, s2]
                                good_springs.append(spring)
                                counter += 1

                                print(f'Springs found: {counter}')

                                # Yield progress every time a spring is found
                                yield iteration, total_iterations, list(good_springs)
        yield iteration, total_iterations, list(good_springs)


# ---- Streamlit App ----
st.title("Duplex rugopar kereso")

# Section: Fixed constraints
st.header("Design feltetelek")

F2 = st.number_input("Nevleges terheles [N]", value=69500)
max_OD = st.number_input("Maximum kulso atmero (D1+d1) [mm]", value=310)
max_height = st.number_input("Nevleges magassag (hezagolas utan) [mm]", value=265)
max_stress = st.number_input("Maximum korrigalatlan solid feszultseg [MPa]", value=650)
min_Sn_s2_gap = st.number_input("Minimum Sn-s2 [mm]", value=10)
min_w = st.number_input("Minimum atmeroviszony D/d", value=4)
max_w = st.number_input("Maximum atmeroviszony D/d", value=9)
stiffness_range = st.slider("Elfogadhato duplex merevseg tartomany [N/mm]", 200, 2000, (700, 950))
G = st.number_input("Nyirasi modulus G [MPa]", value=78500)

# Section: Param Ranges Inputs
st.header("Keresesi tartomany")

min_val_D1, max_val_D1 = st.slider("Kulso rugo D1 [mm]", min_value=100, max_value=400, value=(245, 265), step=1)
min_val_D2, max_val_D2 = st.slider("Belso rugo D2 [mm]", min_value=100, max_value=400, value=(150, 170), step=1)
min_val_n1, max_val_n1 = st.slider("Kulso rugo n1", min_value=1.0, max_value=20.0, value=(3.0, 5.0), step=0.1)
min_val_n2, max_val_n2 = st.slider("Belso rugo n2", min_value=1.0, max_value=20.0, value=(5.0, 7.0), step=0.1)
min_val_L0, max_val_L0 = st.slider("Szabad magassag L0 [mm]", min_value=150, max_value=700, value=(320, 360), step=1)

param_ranges = {
    'd1': ParameterRange('d1', values=[40, 42, 45, 48, 50, 52]),
    'd2': ParameterRange('d2', values=[26, 27, 28, 30, 32, 35, 36, 38, 40, 42, 45]),
    'D1': ParameterRange('D1', start=min_val_D1, stop=max_val_D1, step=1),
    'D2': ParameterRange('D2', start=min_val_D2, stop=max_val_D2, step=1),
    'n1': ParameterRange('n1', start=min_val_n1, stop=max_val_n1, step=0.1),
    'n2': ParameterRange('n2', start=min_val_n2, stop=max_val_n2, step=0.1),
    'L0': ParameterRange('L0', start=min_val_L0, stop=max_val_L0, step=1),
}


# ---- Run the Spring Finder ----
# Setup progress bar and status
progress = st.progress(0)
status = st.empty()

if st.button("Rugok keresese"):
    for iter, total, partial_results in find_springs(
        F2=F2,
        max_OD=max_OD,
        max_height=max_height,
        max_stress=max_stress,
        min_Sn_s2_gap=min_Sn_s2_gap,
        min_w=min_w,
        max_w=max_w,
        stiffness_range=stiffness_range,
        param_ranges=param_ranges,
        G=G
    ):
        good_springs = partial_results  # Updated list
        #percent = int(iter / total * 100)
        #progress.progress(percent)
        status.text(f"Turelem... {len(good_springs)} rugot talaltam eddig.")


    st.success(f"{len(good_springs)} megfelelo rugot talaltam")
    headers = ['L0', 'Lc1', 'Lc2', 'd1', 'd2', 'D1', 'D2', 'n1', 'n2', 'R1', 'R2', 'tauc1', 'tauc2', 'w1', 'w2', 'Sn1',
               'Sn2', 'L2', 's2']
    df = pd.DataFrame(good_springs, columns=headers)
    st.subheader("Talalt Rugok")
    st.dataframe(df, use_container_width=True)

    # Optional: Download as CSV
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="rugok.csv", mime="text/csv")



# # Print results
# wb = Workbook()
# ws = wb.active
#
# headers = ['L0', 'Lc1', 'Lc2', 'd1', 'd2', 'D1', 'D2', 'n1', 'n2', 'R1', 'R2', 'tauc1', 'tauc2', 'w1', 'w2', 'Sn1', 'Sn2', 'L2', 's2']
#
# # Write data to the worksheet
# ws.append(headers)
# for spring in good_springs:
#     ws.append(spring)
#
# # Save the workbook to a file
# wb.save(r'C:\Users\prife\Documents\GANZ\output.xlsx')