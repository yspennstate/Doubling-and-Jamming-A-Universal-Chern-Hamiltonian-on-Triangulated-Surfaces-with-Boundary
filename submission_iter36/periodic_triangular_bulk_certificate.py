import argparse
from pathlib import Path

import numpy as np


GRIDS = [11, 21, 31, 51]
EXPECTED_CHERN = {"H_+": -1, "H_-": 1}
TABLE_PATH = Path(__file__).with_name("periodic_triangular_certificate_table.tex")
EXTENSION_TABLE_PATH = Path(__file__).with_name(
    "periodic_triangular_extension_certificate_table.tex"
)


def triangular_parts(k1, k2):
    z1 = np.exp(1j * k1)
    z2 = np.exp(1j * k2)

    d1 = np.array([[z1 - 1, z2 - 1, z1 * z2 - 1]], dtype=complex)
    d2 = np.array([[1, -z2], [z1, -1], [-1, 1]], dtype=complex)

    B = np.zeros((6, 6), dtype=complex)
    B[0:1, 1:4] = d1
    B[1:4, 4:6] = d2

    P0 = np.array([[np.conj(z1 * z2)], [np.conj(z2)]], dtype=complex)
    P1 = np.array(
        [[0, np.conj(z1), 0], [0, 0, 0], [-np.conj(z2), 0, 0]], dtype=complex
    )
    P2 = np.array([[1, 1]], dtype=complex)

    S = np.zeros((6, 6), dtype=complex)
    S[4:6, 0:1] = 0.5j * (P2.conj().T + P0)
    S[1:4, 1:4] = 0.5j * (P1.conj().T - P1)
    S[0:1, 4:6] = -0.5j * (P0.conj().T + P2)

    return B, S


def triangular_symbol(k1, k2, sign=1, mass=1.0):
    B, S = triangular_parts(k1, k2)
    return B + B.conj().T + sign * mass * S


def pullback_symbol(matrix, sign=1, mass=1.0):
    matrix = np.asarray(matrix, dtype=int)

    def h(k1, k2):
        kk = matrix @ np.array([k1, k2], dtype=float)
        return triangular_symbol(kk[0], kk[1], sign=sign, mass=mass)

    return h


def triangular_hopping_terms(sign=1, mass=1.0):
    n = 7
    terms = {}
    for d1 in range(-2, 3):
        for d2 in range(-2, 3):
            coeff = np.zeros((6, 6), dtype=complex)
            for i in range(n):
                for j in range(n):
                    k1 = 2 * np.pi * i / n
                    k2 = 2 * np.pi * j / n
                    coeff += triangular_symbol(k1, k2, sign=sign, mass=mass) * np.exp(
                        -1j * (d1 * k1 + d2 * k2)
                    )
            coeff /= n * n
            coeff[np.abs(coeff) < 1e-12] = 0.0
            if np.linalg.norm(coeff) > 1e-10:
                terms[(d1, d2)] = coeff
    return terms


def supercell_symbol(L1, L2, sign=1, mass=1.0):
    terms = triangular_hopping_terms(sign=sign, mass=mass)
    norb = 6
    dim = norb * L1 * L2

    def idx(x, y, a):
        return ((x * L2 + y) * norb) + a

    def h(q1, q2):
        out = np.zeros((dim, dim), dtype=complex)
        for x in range(L1):
            for y in range(L2):
                for (d1, d2), coeff in terms.items():
                    tx = x + d1
                    ty = y + d2
                    wx = math_floor_div(tx, L1)
                    wy = math_floor_div(ty, L2)
                    xp = tx % L1
                    yp = ty % L2
                    phase = np.exp(1j * (wx * q1 + wy * q2))
                    for a in range(norb):
                        row0 = idx(xp, yp, a)
                        for b in range(norb):
                            val = coeff[a, b]
                            if val != 0:
                                out[row0, idx(x, y, b)] += val * phase
        return 0.5 * (out + out.conj().T)

    return h


def math_floor_div(a, b):
    return int(np.floor(a / b))


def fhs_chern_for_symbol(n, h_func):
    occ = [[None for _ in range(n)] for _ in range(n)]
    min_gap = float("inf")
    for i in range(n):
        for j in range(n):
            h = h_func(2 * np.pi * i / n, 2 * np.pi * j / n)
            eigvals, eigvecs = np.linalg.eigh(h)
            min_gap = min(min_gap, float(np.min(np.abs(eigvals))))
            occ[i][j] = eigvecs[:, eigvals < 0]
            if occ[i][j].shape[1] != h.shape[0] // 2:
                raise RuntimeError(f"occupied rank changed at {(i, j)}")

    total = 0.0
    min_link_abs = float("inf")
    max_plaquette_abs = 0.0
    for i in range(n):
        for j in range(n):
            u = occ[i][j]
            ux = occ[(i + 1) % n][j]
            uy = occ[i][(j + 1) % n]
            uxy = occ[(i + 1) % n][(j + 1) % n]

            u1 = np.linalg.det(u.conj().T @ ux)
            u2 = np.linalg.det(u.conj().T @ uy)
            u1y = np.linalg.det(uy.conj().T @ uxy)
            u2x = np.linalg.det(ux.conj().T @ uxy)

            min_link_abs = min(
                min_link_abs, abs(u1), abs(u2), abs(u1y), abs(u2x)
            )
            u1 /= abs(u1)
            u2 /= abs(u2)
            u1y /= abs(u1y)
            u2x /= abs(u2x)
            phase = float(np.angle(u1 * u2x / (u1y * u2)))
            max_plaquette_abs = max(max_plaquette_abs, abs(phase))
            total += phase

    return total / (2 * np.pi), min_gap, min_link_abs, max_plaquette_abs


def fhs_chern(n, sign):
    return fhs_chern_for_symbol(
        n, lambda k1, k2: triangular_symbol(k1, k2, sign=sign)
    )


def determinant_gap_formula(k1, k2):
    value = -28 + 8 * np.cos(k1) + 6 * np.cos(k2) + 6 * np.cos(k1 + k2)
    return -(value * value) / 16


def build_rows():
    rows = []
    for sign, name in [(1, "H_+"), (-1, "H_-")]:
        for n in GRIDS:
            chern, min_gap, min_link_abs, max_phase = fhs_chern(n, sign)
            rows.append((name, n, chern, min_gap, min_link_abs, max_phase))
    return rows


def build_extension_rows():
    diagnostics = [
        (
            "positive mass",
            r"$H_{+,m}$, $m=0.5$",
            31,
            -1,
            lambda k1, k2: triangular_symbol(k1, k2, sign=1, mass=0.5),
        ),
        (
            "positive mass",
            r"$H_{-,m}$, $m=2$",
            31,
            1,
            lambda k1, k2: triangular_symbol(k1, k2, sign=-1, mass=2.0),
        ),
        (
            "finite cover",
            r"$H_{+,A}$, $A=\mathrm{diag}(2,1)$",
            31,
            -2,
            pullback_symbol([[2, 0], [0, 1]], sign=1),
        ),
        (
            "finite cover",
            r"$H_{-,A}$, $A=\mathrm{diag}(2,1)$",
            31,
            2,
            pullback_symbol([[2, 0], [0, 1]], sign=-1),
        ),
        (
            "orientation-reversing cover",
            r"$H_{+,A}$, $A=\mathrm{diag}(1,-1)$",
            31,
            1,
            pullback_symbol([[1, 0], [0, -1]], sign=1),
        ),
        (
            "cell enlargement",
            r"$H_+$, $2\times1$ folded cell",
            21,
            -1,
            supercell_symbol(2, 1, sign=1),
        ),
        (
            "cell enlargement",
            r"$H_-$, $2\times1$ folded cell",
            21,
            1,
            supercell_symbol(2, 1, sign=-1),
        ),
    ]
    rows = []
    for kind, model, n, expected, h_func in diagnostics:
        chern, min_gap, min_link_abs, max_phase = fhs_chern_for_symbol(n, h_func)
        rows.append((kind, model, n, expected, chern, min_gap, min_link_abs, max_phase))
    return rows


def validate_rows(rows, integer_tol=5e-8, min_link_threshold=1e-3, branch_margin=1e-3):
    for name, n, chern, min_gap, min_link_abs, max_phase in rows:
        rounded = round(chern)
        if abs(chern - rounded) > integer_tol:
            raise RuntimeError(f"{name} {n}x{n}: FHS value is not integer: {chern}")
        if rounded != EXPECTED_CHERN[name]:
            raise RuntimeError(
                f"{name} {n}x{n}: expected {EXPECTED_CHERN[name]}, got {rounded}"
            )
        if min_gap <= 0:
            raise RuntimeError(f"{name} {n}x{n}: sampled zero gap vanished")
        if min_link_abs <= min_link_threshold:
            raise RuntimeError(
                f"{name} {n}x{n}: link determinant too small: {min_link_abs}"
            )
        if max_phase >= np.pi - branch_margin:
            raise RuntimeError(
                f"{name} {n}x{n}: plaquette phase too close to branch cut: {max_phase}"
            )


def validate_extension_rows(
    rows, integer_tol=5e-8, min_link_threshold=1e-4, branch_margin=1e-3
):
    for kind, model, n, expected, chern, min_gap, min_link_abs, max_phase in rows:
        rounded = round(chern)
        if abs(chern - rounded) > integer_tol:
            raise RuntimeError(f"{kind} {model}: FHS value is not integer: {chern}")
        if rounded != expected:
            raise RuntimeError(f"{kind} {model}: expected {expected}, got {rounded}")
        if min_gap <= 0:
            raise RuntimeError(f"{kind} {model}: sampled zero gap vanished")
        if min_link_abs <= min_link_threshold:
            raise RuntimeError(f"{kind} {model}: link determinant too small: {min_link_abs}")
        if max_phase >= np.pi - branch_margin:
            raise RuntimeError(
                f"{kind} {model}: plaquette phase too close to branch cut: {max_phase}"
            )


def write_table(rows):
    with TABLE_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{llrrrr}\\toprule\n")
        fh.write(
            "Hamiltonian & Grid & FHS Chern & min $|E|$ & min link & max plaquette \\\\\n"
        )
        fh.write("\\midrule\n")
        for name, n, chern, min_gap, min_link_abs, max_phase in rows:
            fh.write(
                f"${name}$ & ${n}\\times{n}$ & {chern:.6f} & "
                f"{min_gap:.6f} & {min_link_abs:.6f} & {max_phase:.6f} \\\\\n"
            )
        fh.write("\\bottomrule\n\\end{tabular}\n")


def write_extension_table(rows):
    with EXTENSION_TABLE_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{llrrrrr}\\toprule\n")
        fh.write(
            "Diagnostic & Model & Grid & expected & FHS & min $|E|$ & min link \\\\\n"
        )
        fh.write("\\midrule\n")
        for kind, model, n, expected, chern, min_gap, min_link_abs, _ in rows:
            fh.write(
                f"{kind} & {model} & ${n}\\times{n}$ & {expected:d} & "
                f"{chern:.6f} & {min_gap:.6f} & {min_link_abs:.6f} \\\\\n"
            )
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and check the periodic triangular FHS Chern certificate."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the certificate values without requiring table output",
    )
    parser.add_argument(
        "--write-table",
        action="store_true",
        help="write periodic_triangular_certificate_table.tex and extension diagnostics",
    )
    args = parser.parse_args()

    if not args.check and not args.write_table:
        args.write_table = True

    rows = build_rows()
    extension_rows = build_extension_rows()
    validate_rows(rows)
    validate_extension_rows(extension_rows)

    if args.write_table:
        write_table(rows)
        write_extension_table(extension_rows)

    for row in rows:
        print(row)
    for row in extension_rows:
        print(row)
    print("All FHS certificate checks passed.")
    print("Exact determinant formula:")
    print("det H_pm(k) = -(-28 + 8 cos k1 + 6 cos k2 + 6 cos(k1+k2))^2 / 16")
    print("Therefore det H_pm(k) <= -4 and ||H_pm(k)|| <= 8 gives gap >= 1/8192.")


if __name__ == "__main__":
    main()
