import json
from pathlib import Path

import contourpy
import numpy as np
from scipy import optimize

MU_0 = 4e-7 * np.pi
# the share of the current driven by F F' against p'; the value of the paper's ITER case
A_PROFILE = -0.155
N_GRID = 257
GRID_MARGIN = 0.15
CONTOUR_GRID = 801
N_THETA = 64
NAME_FORMAT = "size{size:.2f}_kappa{kappa:.2f}_delta{delta:+.2f}"


def homogeneous(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L = np.log(x)
    x2, x3, x4, x5, x6 = x**2, x**3, x**4, x**5, x**6
    y2, y3, y4, y5, y6 = y**2, y**3, y**4, y**5, y**6
    one, zero = np.ones_like(x), np.zeros_like(x)
    value = np.stack([
        one, x2, y2 - x2 * L, x4 - 4 * x2 * y2,
        2 * y4 - 9 * y2 * x2 + 3 * x4 * L - 12 * x2 * y2 * L,
        x6 - 12 * x4 * y2 + 8 * x2 * y4,
        8 * y6 - 140 * y4 * x2 + 75 * y2 * x4 - 15 * x6 * L + 180 * x4 * y2 * L
        - 120 * x2 * y4 * L])
    d_x = np.stack([
        zero, 2 * x, -2 * x * L - x, 4 * x3 - 8 * x * y2,
        12 * x3 * L + 3 * x3 - 30 * x * y2 - 24 * x * y2 * L,
        6 * x5 - 48 * x3 * y2 + 16 * x * y4,
        -400 * x * y4 + 480 * x3 * y2 - 90 * x5 * L - 15 * x5 + 720 * x3 * y2 * L
        - 240 * x * y4 * L])
    d_xx = np.stack([
        zero, 2 * one, -2 * L - 3, 12 * x2 - 8 * y2,
        36 * x2 * L + 21 * x2 - 54 * y2 - 24 * y2 * L,
        30 * x4 - 144 * x2 * y2 + 16 * y4,
        -640 * y4 + 2160 * x2 * y2 - 450 * x4 * L - 165 * x4 + 2160 * x2 * y2 * L
        - 240 * y4 * L])
    d_y = np.stack([
        zero, zero, 2 * y, -8 * x2 * y,
        8 * y3 - 18 * y * x2 - 24 * x2 * y * L,
        -24 * x4 * y + 32 * x2 * y3,
        48 * y5 - 560 * y3 * x2 + 150 * y * x4 + 360 * x4 * y * L - 480 * x2 * y3 * L])
    d_yy = np.stack([
        zero, zero, 2 * one, -8 * x2,
        24 * y2 - 18 * x2 - 24 * x2 * L,
        -24 * x4 + 96 * x2 * y2,
        240 * y4 - 1680 * y2 * x2 + 150 * x4 + 360 * x4 * L - 1440 * x2 * y2 * L])
    return {"v": value, "x": d_x, "xx": d_xx, "y": d_y, "yy": d_yy}


def particular(x, y, a_profile):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L = np.log(x)
    zero = np.zeros_like(x)
    value = x**4 / 8 + a_profile * (x**2 * L / 2 - x**4 / 8)
    d_x = x**3 / 2 + a_profile * (x * L + x / 2 - x**3 / 2)
    d_xx = 3 * x**2 / 2 + a_profile * (L + 1.5 - 3 * x**2 / 2)
    return {"v": value, "x": d_x, "xx": d_xx, "y": zero, "yy": zero}


def coefficients(eps, kappa, delta, a_profile=A_PROFILE):
    alpha = np.arcsin(delta)
    n1 = -(1 + alpha) ** 2 / (eps * kappa**2)
    n2 = (1 - alpha) ** 2 / (eps * kappa**2)
    n3 = -kappa / (eps * np.cos(alpha) ** 2)
    xo, xi, xt, yt = 1 + eps, 1 - eps, 1 - delta * eps, kappa * eps
    rows = [
        ((xo, 0.0), "v", None, 0.0), ((xi, 0.0), "v", None, 0.0),
        ((xt, yt), "v", None, 0.0), ((xt, yt), "x", None, 0.0),
        ((xo, 0.0), "yy", "x", n1), ((xi, 0.0), "yy", "x", n2),
        ((xt, yt), "xx", "y", n3),
    ]
    matrix, rhs = np.zeros((7, 7)), np.zeros(7)
    for r, ((px, py), first, second, weight) in enumerate(rows):
        h, p = homogeneous(px, py), particular(px, py, a_profile)
        matrix[r] = h[first] + (weight * h[second] if second else 0.0)
        rhs[r] = -(p[first] + (weight * p[second] if second else 0.0))
    return np.linalg.solve(matrix, rhs)


def evaluate(x, y, coeffs, a_profile, key="v"):
    h, p = homogeneous(x, y), particular(x, y, a_profile)
    return p[key] + np.tensordot(coeffs, h[key], axes=(0, 0))


def closed_contour(generator, level, x_inside):
    # the closed contour at this level that encloses the axis
    pieces = generator.create_contour(level)
    for piece in pieces:
        px, py = piece[:, 0], piece[:, 1]
        if not (np.allclose(piece[0], piece[-1], atol=1e-9)):
            continue
        if px.min() < x_inside < px.max() and py.min() < 0.0 < py.max():
            return px[:-1], py[:-1]
    raise ValueError(f"no closed flux surface at level {level}")


def line_integral(px, py, values):
    # trapezoid rule around a closed polygon
    dx = np.roll(px, -1) - px
    dy = np.roll(py, -1) - py
    dl = np.hypot(dx, dy)
    return float(np.sum(0.5 * (values + np.roll(values, -1)) * dl))


class Equilibrium:
    def __init__(self, R_major, a_minor, B_0, Ip, kappa, delta, a_profile=A_PROFILE):
        self.R_major, self.a_minor, self.B_0, self.Ip = R_major, a_minor, B_0, Ip
        self.kappa, self.delta, self.a_profile = kappa, delta, a_profile
        self.eps = a_minor / R_major
        self.coeffs = coefficients(self.eps, kappa, delta, a_profile)
        found = optimize.minimize_scalar(
            lambda x: float(self.psi_norm(x, 0.0)),
            bounds=(1 - self.eps + 1e-3, 1 + self.eps - 1e-3), method="bounded",
            options={"xatol": 1e-10})
        self.x_axis, self.psi_axis = float(found.x), float(found.fun)
        if self.psi_axis >= 0.0:
            raise ValueError("the axis is not below the boundary flux; the shape is not "
                             "a nested equilibrium")
        margin = 0.5 * self.eps
        xs = np.linspace(1 - self.eps - margin, 1 + self.eps + margin, CONTOUR_GRID)
        ys = np.linspace(-(kappa * self.eps + margin), kappa * self.eps + margin,
                         CONTOUR_GRID)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        self.generator = contourpy.contour_generator(
            grid_x, grid_y, self.psi_norm(grid_x, grid_y))
        bx, by = self.contour(0.0)
        gx, gy = self.gradient(bx, by)
        current_integral = line_integral(bx, by, np.hypot(gx, gy) / bx)
        self.psi_scale = 2 * np.pi * MU_0 * R_major * Ip / current_integral
        self.boundary = (bx, by)
        self.psi_axis_wb = self.psi_scale * self.psi_axis

    def psi_norm(self, x, y):
        return evaluate(x, y, self.coeffs, self.a_profile, "v")

    def gradient(self, x, y):
        return (evaluate(x, y, self.coeffs, self.a_profile, "x"),
                evaluate(x, y, self.coeffs, self.a_profile, "y"))

    def contour(self, level):
        return closed_contour(self.generator, level, self.x_axis)

    def f_of_psi(self, psi_wb):
        # F F' is the constant -A Psi0 / (2 pi R0)^2 in Wb units, F = R0 B0 at the boundary
        ff_prime = -self.a_profile * self.psi_scale / ((2 * np.pi) ** 2 * self.R_major**2)
        return np.sqrt((self.R_major * self.B_0) ** 2 + 2 * ff_prime * np.asarray(psi_wb))

    def p_prime(self):
        return -(1 - self.a_profile) * self.psi_scale / (
            (2 * np.pi) ** 2 * MU_0 * self.R_major**4)

    def safety_factor(self, psi_wb):
        psi_wb = np.atleast_1d(np.asarray(psi_wb, dtype=float))
        out = np.empty(psi_wb.size)
        f_values = self.f_of_psi(psi_wb)
        for k, level in enumerate(psi_wb / self.psi_scale):
            cx, cy = self.contour(level)
            gx, gy = self.gradient(cx, cy)
            out[k] = (f_values[k] * self.R_major / self.psi_scale
                      * line_integral(cx, cy, 1.0 / (cx * np.hypot(gx, gy))))
        return out

    def surfaces(self, psi_wb, n_theta=N_THETA):
        theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        psi_wb = np.asarray(psi_wb, dtype=float)
        out = {k: np.empty((psi_wb.size, n_theta)) for k in ("R", "Z", "B", "B_pol")}
        for k, level_wb in enumerate(psi_wb):
            level = level_wb / self.psi_scale
            if level <= self.psi_axis * (1 - 1e-6):
                x = np.full(n_theta, self.x_axis)
                y = np.zeros(n_theta)
            else:
                cx, cy = self.contour(level)
                angle = np.arctan2(cy, cx - self.x_axis) % (2 * np.pi)
                order = np.argsort(angle)
                angle, cx, cy = angle[order], cx[order], cy[order]
                wrap = np.concatenate([angle[-1:] - 2 * np.pi, angle, angle[:1] + 2 * np.pi])
                x = np.interp(theta, wrap, np.concatenate([cx[-1:], cx, cx[:1]]))
                y = np.interp(theta, wrap, np.concatenate([cy[-1:], cy, cy[:1]]))
            gx, gy = self.gradient(x, y)
            radius = self.R_major * x
            b_pol = self.psi_scale * np.hypot(gx, gy) / (2 * np.pi * self.R_major**2 * x)
            b_tor = self.f_of_psi(level_wb) / radius
            out["R"][k], out["Z"][k] = radius, self.R_major * y
            out["B_pol"][k], out["B"][k] = b_pol, np.hypot(b_pol, b_tor)
        return theta, out

    def eqdsk_dict(self, name):
        R0, a, kappa = self.R_major, self.a_minor, self.kappa
        margin = GRID_MARGIN * R0 / 1.67
        r_lo, r_hi = R0 * (1 - self.eps) - margin, R0 * (1 + self.eps) + margin
        z_hi = kappa * a + margin
        r_1d = np.linspace(r_lo, r_hi, N_GRID)
        z_1d = np.linspace(-z_hi, z_hi, N_GRID)
        grid_r, grid_z = np.meshgrid(r_1d, z_1d, indexing="ij")
        psi_2d = self.psi_scale * self.psi_norm(grid_r / R0, grid_z / R0)
        psi_1d = np.linspace(self.psi_axis_wb, 0.0, N_GRID)
        q = np.empty(N_GRID)
        q[1:] = self.safety_factor(psi_1d[1:])
        # the axis carries no contour: quadratic extrapolation from the first surfaces
        q[0] = np.polyval(np.polyfit(psi_1d[1:4], q[1:4], 2), psi_1d[0])
        bx, by = self.boundary
        limiter_r = np.array([r_lo, r_hi, r_hi, r_lo, r_lo]) + np.array([1, -1, -1, 1, 1]) * 0.01
        limiter_z = np.array([-z_hi, -z_hi, z_hi, z_hi, -z_hi]) + np.array([1, 1, -1, -1, 1]) * 0.01
        return {
            "name": name, "nx": N_GRID, "nz": N_GRID,
            "xdim": r_hi - r_lo, "zdim": 2 * z_hi, "xcentre": R0, "xgrid1": r_lo,
            "zmid": 0.0, "xmag": R0 * self.x_axis, "zmag": 0.0,
            "psimag": self.psi_axis_wb, "psibdry": 0.0, "bcentre": self.B_0,
            "cplasma": self.Ip,
            "fpol": self.f_of_psi(psi_1d), "pressure": self.p_prime() * psi_1d,
            "ffprime": np.full(N_GRID, -self.a_profile * self.psi_scale
                               / ((2 * np.pi) ** 2 * R0**2)),
            "pprime": np.full(N_GRID, self.p_prime()), "psi": psi_2d, "qpsi": q,
            "nbdry": bx.size + 1,
            "xbdry": R0 * np.append(bx, bx[0]), "zbdry": R0 * np.append(by, by[0]),
            "nlim": limiter_r.size, "xlim": limiter_r, "zlim": limiter_z,
            "ncoil": 0, "xc": np.zeros(0), "zc": np.zeros(0), "dxc": np.zeros(0),
            "dzc": np.zeros(0), "Ic": np.zeros(0),
            "x": r_1d, "z": z_1d, "psinorm": np.linspace(0.0, 1.0, N_GRID),
        }

    def summary(self):
        psi_1d = np.linspace(self.psi_axis_wb, 0.0, 41)
        q = self.safety_factor(psi_1d[1:])
        return {"R_major": self.R_major, "a_minor": self.a_minor, "B_0": self.B_0,
                "Ip": self.Ip, "kappa": self.kappa, "delta": self.delta,
                "a_profile": self.a_profile, "psi_scale_wb": self.psi_scale,
                "psi_axis_wb": self.psi_axis_wb, "R_axis": self.R_major * self.x_axis,
                "q_near_axis": float(q[0]), "q_boundary": float(q[-1]),
                "pressure_axis_pa": float(self.p_prime() * self.psi_axis_wb)}


def shape_name(size, kappa, delta):
    return NAME_FORMAT.format(size=size, kappa=kappa, delta=delta)


def build(machine, size, kappa, delta):
    # the machine scales with size at fixed aspect ratio, field and safety factor
    return Equilibrium(R_major=machine["R_major"] * size, a_minor=machine["a_minor"] * size,
                       B_0=machine["B_0"], Ip=machine["Ip"] * size, kappa=kappa, delta=delta)


def write_eqdsk(equilibrium, path, name):
    import eqdsk

    data = equilibrium.eqdsk_dict(name)
    interface = eqdsk.EQDSKInterface.from_dict(data, no_cocos=True)
    interface.write(str(path), file_format="eqdsk")


def ensure_library(directory, machine, shapes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "library.json"
    index = json.load(open(index_path)) if index_path.is_file() else {}
    for size, kappa, delta in shapes:
        name = shape_name(size, kappa, delta)
        path = directory / f"{name}.eqdsk"
        if path.is_file() and name in index:
            continue
        equilibrium = build(machine, size, kappa, delta)
        write_eqdsk(equilibrium, path, name)
        index[name] = dict(equilibrium.summary(), size=size, file=path.name)
        with open(index_path, "w") as handle:
            json.dump(index, handle, indent=1)
    return index


def psi_of_rho_levels(equilibrium, rho, last_surface_factor=0.99, n_levels=121):
    psi_last = equilibrium.psi_axis_wb * (1.0 - last_surface_factor)
    levels = np.linspace(equilibrium.psi_axis_wb, psi_last, n_levels)
    q = np.empty(n_levels)
    q[1:] = equilibrium.safety_factor(levels[1:])
    q[0] = np.polyval(np.polyfit(levels[1:4], q[1:4], 2), levels[0])
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(levels))])
    rho_of_levels = np.sqrt(phi / phi[-1])
    return np.interp(np.asarray(rho, dtype=float), rho_of_levels, levels)


Equilibrium.psi_of_rho = lambda self, rho: psi_of_rho_levels(self, rho)
