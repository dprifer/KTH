from openpyxl import Workbook


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
    good_springs = []
    counter = 0
    for L0 in param_ranges['L0'].get_values():
        print(f'Spring: L0={L0}')
        for d1 in param_ranges['d1'].get_values():
            for D1 in param_ranges['D1'].get_values():
                if D1 + d1 > max_OD:
                    continue
                for n1 in param_ranges['n1'].get_values():
                    for d2 in param_ranges['d2'].get_values():
                        for D2 in param_ranges['D2'].get_values():
                            for n2 in param_ranges['n2'].get_values():
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

                                if not (Sn1-min_Sn_s2_gap > s2 and Sn2-min_Sn_s2_gap > s2 and L2 < max_height and min_w < w1 < max_w and min_w < w2 < max_w and tauc1 < max_stress and tauc2 < max_stress):
                                    continue

                                spring = [L0, Lc1, Lc2, d1, d2, D1, D2, n1, n2, R1, R2, tauc1, tauc2, w1, w2, Sn1, Sn2, L2, s2]

                                good_springs.append(spring)
                                counter += 1
                                print(f'Springs found: {counter}')

                                #print(f'L0={L0}, d1 = {d1}, D1 = {D1}, n1 = {n1}, d2={d2}, D2={D2}, n2 = {n2}, Sn1 = {Sn1}, Sn2 = {Sn2}, Lc1 = {Lc1}, Lc2 = {Lc2}, tauc1 = {tauc1}, tauc2 = {tauc2}')
    return good_springs


param_ranges = {
    'd1': ParameterRange('d1', values=[40, 42, 45, 48, 50, 52]),  # wire diameter outer
    'd2': ParameterRange('d1', values=[26, 27, 28, 30, 32, 35, 36, 38, 40, 42, 45]),  # wire diameter inner
    'D1': ParameterRange('D1', start=245, stop=265, step=1), # mean diameter outer
    'D2': ParameterRange('D2', start=150, stop=170, step=1), # mean diameter inner
    'n1': ParameterRange('n1', start=3, stop=5, step=0.1),        # active coils outer
    'n2': ParameterRange('n2', start=5, stop=7, step=0.1),        # active coils inner
    'L0': ParameterRange('L0', start=320, stop=360, step=1),        # free length inner
}

# Call the search function
good_springs = find_springs(
    F2=69500,
    max_OD=310,
    max_height = 265,
    max_stress = 650,
    min_Sn_s2_gap = 10,
    min_w = 4,
    max_w = 9,
    stiffness_range=(700, 950),
    param_ranges=param_ranges,
    G=78500 # shear modulus (MPa)
)

# Print results
wb = Workbook()
ws = wb.active

headers = ['L0', 'Lc1', 'Lc2', 'd1', 'd2', 'D1', 'D2', 'n1', 'n2', 'R1', 'R2', 'tauc1', 'tauc2', 'w1', 'w2', 'Sn1', 'Sn2', 'L2', 's2']

# Write data to the worksheet
ws.append(headers)
for spring in good_springs:
    ws.append(spring)

# Save the workbook to a file
wb.save(r'C:\Users\prife\Documents\GANZ\output.xlsx')