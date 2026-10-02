import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192e-27

CLOSURE = dict(
    w_ped=0.05,                  # pedestal width in normalised poloidal flux
    n_core_over_ped=1.6,         # the prescribed density profile, after the OFT example
    n_sep_over_ped=0.32,
    beta_p_ped_H=0.35,           # the pedestal a barrier holds: beta_p_ped_H * <Bp>^2 / 2 mu0,
                                 # <Bp> = mu0 Ip over the poloidal perimeter of the boundary
    coil_weight=0.2,             # how hard the coil currents are pulled to the reference discharge
    coil_weight_fallback=5.0,    # the pull is multiplied by this when the surfaces come out not nested
    coil_bound_factor=4.0,       # coil currents bounded at this multiple of the reference discharge's
    isoflux_weight=5.0,          # weight of every boundary point and of the X-point in the fit
    chi_L=1.0,                   # m^2/s, the diffusivity of the barrier-free edge
    psi_pedestal=0.8,            # the pedestal region GPEC is read over
    access_beta_p=0.12,          # the pedestal the unstable band is read on, below the first
    psi_band=0.95,               # the steep part of the pedestal the unstable band is read over
    band_max=7.5,                # the barrier is entered where the second boundary sits below
    barrier_ladder=(0.85, 0.7),  # fractions of the barrier pedestal tried when the equilibrium
                                 # does not hold the full one
    f_outer=0.6,                 # share of P_SOL reaching the outer target
    lambda_eich=(0.63, -1.19),   # lambda_q [mm] = 0.63 Bp_mid^-1.19, Eich 2013 regression 14
    c_imp=(1.0, 2.0),            # tau_imp/tau_E = c0 + c1 barrier max(p_ped/p_ped_L - 1, 0)
    psi_lcfs=0.995,              # the closed surface metrics are read on
    psi_sol=1.005,               # the open surface the divertor geometry is read on
    beta_N_seed=1.5,             # the pressure of the first equilibrium of a setting
    P_SOL_guess=0.7,             # P_SOL / P_heat before the first TORAX run
    Z_imp=6.0,                   # carbon
    A_i=2.0,                     # deuterium
)
SPEC_REQUIRED = ("R0", "a", "kappa", "delta", "Ip", "B0", "n_e_ped", "P_heat")
SPEC_DEFAULTS = dict(Z0=0.0, kappa_lower=None, delta_lower=None, machine_scale=1.0, Z_eff=2.0,
                     impurity="C", T_sep=100.0, heating_location=0.0, heating_width=0.3,
                     electron_heat_fraction=0.5)
IMPURITY_CHARGE = {"C": 6.0, "N": 7.0, "O": 8.0, "Ne": 10.0, "Ar": 18.0}
TORAX = dict(n_rho=25, t_final=0.5, t_final_first=0.3, max_dt=0.05, transport="qlknn", n_surfaces=100,
             last_surface_factor=0.99, cocos=7, chi_min=0.05, chi_max=10.0, D_e_min=0.05)
RETRY_MAXITS = 500                   # the second try of a solve, from scratch
RETRY_URF = 0.1                      # with half the under-relaxation of TokaMaker's default
NUDGE_FACTORS = (0.9, 1.1)           # nearby pedestal heights tried when a solve fails at one
BARRIER_LADDER_LOW = (0.55, 0.45)    # further fractions of the barrier pedestal, for the weakly
N_PSI = 201
K_THETA = 64
MAP_POINTS = 129                     # the flux map the mechanism is given, per side
RING_POINTS = 128                    # the boundary and the limiter it is given, points each
RHO_ALPHA = 0.95
FINAL_FILES = ("{name}_eq.geqdsk", "{name}_eq.json", "{name}_run.nc", "{name}.json")


def complete_spec(spec):
    out = dict(SPEC_DEFAULTS)
    out.update(spec)
    missing = [key for key in SPEC_REQUIRED if key not in out]
    if missing:
        raise ValueError(f"setting lacks {missing}")
    unknown = sorted(set(out) - set(SPEC_DEFAULTS) - set(SPEC_REQUIRED))
    if unknown:
        raise ValueError(f"unknown setting keys {unknown}; accepted "
                         f"{list(SPEC_REQUIRED) + list(SPEC_DEFAULTS)}")
    if out["kappa_lower"] is None:
        out["kappa_lower"] = out["kappa"]
    if out["delta_lower"] is None:
        out["delta_lower"] = out["delta"]
    if str(out["impurity"]) not in IMPURITY_CHARGE:
        raise ValueError(f"impurity is one of {list(IMPURITY_CHARGE)}")
    return {key: (str(value) if key == "impurity" else float(value)) for key, value in out.items()}


def composition(z_eff, z_imp):
    imp_frac = float(np.clip((z_eff - 1.0) / (z_imp * (z_imp - 1.0)), 0.0, 1.0 / z_imp))
    n_i_over_n_e = 1.0 - z_imp * imp_frac
    return n_i_over_n_e, imp_frac, 1.0 + n_i_over_n_e + imp_frac


def pedestal_profile(edge, ped, core, x, expin=1.5, expout=1.5, width=0.05):
    # the OMFIT H-mode profile form the OFT bootstrap module uses, on any grid x in [0, 1]
    x = np.asarray(x, dtype=float)
    w_e1 = 0.5 * width
    xphalf = 1.0 - w_e1
    xped = xphalf - w_e1
    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_e1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)
    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_e1) - pconst) + edge
    val = 0.5 * a_t * (1.0 - np.tanh((x - xphalf) / w_e1) - pconst) + edge
    xtoped = x / xped
    inside = xtoped ** expin < 1.0
    val = np.where(inside, val + (core - coretanh) * np.clip(1.0 - xtoped ** expin, 0.0, None)
                   ** expout, val)
    return val


# ---- gEQDSK and the geometry read from it -------------------------------------------

def read_geqdsk(path):
    with open(path) as handle:
        lines = handle.read().splitlines()
    head = lines[0].split()
    nw, nh = int(head[-2]), int(head[-1])
    tokens = []
    for line in lines[1:]:
        if "." not in line:
            tokens += [int(v) for v in line.split()]
        else:
            tokens += [float(v.replace("D", "e").replace("d", "e")) for v in
                       re.findall(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?", line)]
    pos = 0

    def take(count):
        nonlocal pos
        out = np.asarray(tokens[pos:pos + count], dtype=float)
        pos += count
        if out.size != count:
            raise ValueError(f"{path} ends early")
        return out

    rdim, zdim, rcentr, rleft, zmid = take(5)
    rmaxis, zmaxis, simag, sibry, bcentr = take(5)
    current = take(5)[0]
    take(5)
    eq = {"nw": nw, "nh": nh, "rdim": rdim, "zdim": zdim, "rcentr": rcentr, "rleft": rleft,
          "zmid": zmid, "rmaxis": rmaxis, "zmaxis": zmaxis, "simag": simag, "sibry": sibry,
          "bcentr": bcentr, "current": current}
    eq["fpol"], eq["pres"], eq["ffprim"], eq["pprime"] = (take(nw) for _ in range(4))
    eq["psirz"] = take(nw * nh).reshape(nh, nw)
    eq["qpsi"] = take(nw)
    nbbbs, limitr = int(tokens[pos]), int(tokens[pos + 1])
    pos += 2
    bbbs = take(2 * nbbbs).reshape(nbbbs, 2)
    lim = take(2 * limitr).reshape(limitr, 2)
    eq["rbbbs"], eq["zbbbs"] = bbbs[:, 0], bbbs[:, 1]
    eq["rlim"], eq["zlim"] = lim[:, 0], lim[:, 1]
    eq["R"] = rleft + rdim * np.arange(nw) / (nw - 1)
    eq["Z"] = zmid - zdim / 2.0 + zdim * np.arange(nh) / (nh - 1)
    eq["psi_n"] = (eq["psirz"] - simag) / (sibry - simag)
    eq["psi_grid"] = np.linspace(0.0, 1.0, nw)
    # the toroidal flux of every surface, from q, gives the normalised toroidal flux radius
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (eq["qpsi"][1:] + eq["qpsi"][:-1])
                                            * np.diff(eq["psi_grid"]))])
    eq["rho_of_psi_grid"] = np.sqrt(np.clip(phi / phi[-1], 0.0, 1.0))
    return eq


def rho_of_psi(eq, psi_n):
    return float(np.interp(psi_n, eq["psi_grid"], eq["rho_of_psi_grid"]))


def psi_of_rho(eq, rho):
    return float(np.interp(rho, eq["rho_of_psi_grid"], eq["psi_grid"]))


def contour_lines(eq, level):
    import contourpy

    generator = contourpy.contour_generator(x=eq["R"], y=eq["Z"], z=eq["psi_n"])
    return [np.asarray(line, dtype=float) for line in generator.lines(float(level))]


def inside_polygon(point, polygon):
    x, y = point
    px, py = polygon[:, 0], polygon[:, 1]
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if (py[i] > y) != (py[j] > y):
            x_cross = px[j] + (y - py[j]) * (px[i] - px[j]) / (py[i] - py[j])
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def closed_surface(eq, level):
    axis = (eq["rmaxis"], eq["zmaxis"])
    best = None
    for line in contour_lines(eq, level):
        if len(line) < 8 or np.hypot(*(line[0] - line[-1])) > 0.05:
            continue
        if inside_polygon(axis, line):
            if best is None or len(line) > len(best):
                best = line
    if best is None:
        raise RuntimeError(f"no closed surface at psi_n = {level}")
    return best


def by_angle(line, axis, k=K_THETA):
    r, z = line[:, 0] - axis[0], line[:, 1] - axis[1]
    theta = np.mod(np.arctan2(z, r), 2.0 * np.pi)
    order = np.argsort(theta)
    theta, r, z = theta[order], r[order], z[order]
    theta_ext = np.concatenate([theta - 2.0 * np.pi, theta, theta + 2.0 * np.pi])
    r_ext, z_ext = np.tile(r, 3), np.tile(z, 3)
    target = 2.0 * np.pi * np.arange(k) / k
    return np.column_stack([axis[0] + np.interp(target, theta_ext, r_ext),
                            axis[1] + np.interp(target, theta_ext, z_ext)])


class Fields:
    def __init__(self, eq):
        dpsi_dz, dpsi_dr = np.gradient(eq["psirz"], eq["Z"], eq["R"])
        self.psi_n = RegularGridInterpolator((eq["Z"], eq["R"]), eq["psi_n"], bounds_error=False,
                                             fill_value=None)
        self.dr = RegularGridInterpolator((eq["Z"], eq["R"]), dpsi_dr, bounds_error=False,
                                          fill_value=None)
        self.dz = RegularGridInterpolator((eq["Z"], eq["R"]), dpsi_dz, bounds_error=False,
                                          fill_value=None)
        self.eq = eq

    def __call__(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        zr = np.column_stack([pts[:, 1], pts[:, 0]])
        b_pol = np.hypot(self.dr(zr), self.dz(zr)) / pts[:, 0]
        psi_n = np.clip(self.psi_n(zr), 0.0, 1.0)
        f = np.interp(psi_n, self.eq["psi_grid"], self.eq["fpol"])
        b_tor = np.abs(f) / pts[:, 0]
        return b_pol, b_tor, np.hypot(b_pol, b_tor), psi_n


def segment_intersection(p, q, a, b):
    d1, d2 = q - p, b - a
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-14:
        return None
    w = a - p
    t = (w[0] * d2[1] - w[1] * d2[0]) / denominator
    u = (w[0] * d1[1] - w[1] * d1[0]) / denominator
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p + t * d1
    return None


def divertor_geometry(eq, fields, r_out_mid, closure=CLOSURE):
    axis = (eq["rmaxis"], eq["zmaxis"])
    target = np.array([r_out_mid + 0.01, axis[1]])
    lines = contour_lines(eq, closure["psi_sol"])
    if not lines:
        raise RuntimeError("no open surface outside the boundary")
    best, best_i, best_d = None, 0, np.inf
    for line in lines:
        d = np.hypot(line[:, 0] - target[0], line[:, 1] - target[1])
        if d.min() < best_d:
            best, best_i, best_d = line, int(d.argmin()), float(d.min())
    line = best
    # walk in the direction that leaves the midplane downwards, towards the lower X-point
    step = -1 if (best_i > 0 and line[best_i - 1, 1] < line[best_i, 1]) else 1
    limiter = np.column_stack([eq["rlim"], eq["zlim"]])
    lim_segments = [(limiter[i], limiter[(i + 1) % len(limiter)]) for i in range(len(limiter))]
    l_par, strike, on_edge = 0.0, None, True
    i = best_i
    b_mid = fields(line[best_i])
    while 0 <= i + step < len(line):
        p, q = line[i], line[i + step]
        hit = None
        for a, b in lim_segments:
            hit = segment_intersection(p, q, a, b)
            if hit is not None:
                break
        end = q if hit is None else hit
        mid = 0.5 * (p + end)
        b_pol, _, b_tot, _ = fields(mid)
        dl = float(np.hypot(*(end - p)))
        l_par += dl * float(b_tot[0] / max(b_pol[0], 1e-6))
        if hit is not None:
            strike, on_edge = hit, False
            break
        i += step
    if strike is None:
        strike = line[i]
    b_s = fields(strike)
    r_strike = float(strike[0])
    bp_mid, bp_strike = float(b_mid[0][0]), float(max(b_s[0][0], 1e-6))
    # the angle, in the poloidal plane, between the leg and the target it meets
    sin_target = 1.0
    if not on_edge:
        leg = line[i + step] - line[i]
        for a, b in lim_segments:
            if segment_intersection(line[i], line[i + step], a, b) is not None:
                plate = b - a
                cross = abs(leg[0] * plate[1] - leg[1] * plate[0])
                sin_target = float(cross / max(np.hypot(*leg) * np.hypot(*plate), 1e-12))
                break
    return {"R_strike": r_strike, "Z_strike": float(strike[1]),
            "f_exp": float(line[best_i, 0] * bp_mid / (r_strike * bp_strike)),
            "B_pol_mid": bp_mid, "L_par": float(l_par), "sin_target": sin_target,
            "strike_on_grid_edge": on_edge}


def shape_of(eq, closure=CLOSURE):
    lcfs = closed_surface(eq, closure["psi_lcfs"])
    r, z = lcfs[:, 0], lcfs[:, 1]
    r0, a, z_axis = 0.5 * (r.max() + r.min()), 0.5 * (r.max() - r.min()), eq["zmaxis"]
    return {"R_geo": float(r0), "a_geo": float(a), "kappaU": float((z.max() - z_axis) / a),
            "kappaL": float((z_axis - z.min()) / a), "deltaU": float((r0 - r[z.argmax()]) / a),
            "deltaL": float((r0 - r[z.argmin()]) / a)}


def shape_error(asked, got):
    pairs = (("a", "a_geo"), ("kappa", "kappaU"), ("kappa_lower", "kappaL"),
             ("delta", "deltaU"), ("delta_lower", "deltaL"))
    return {stats_key: round(float(got[stats_key]) - float(asked[spec_key]), 3)
            for spec_key, stats_key in pairs}


def geometry_of(eq, closure=CLOSURE):
    fields = Fields(eq)
    axis = (eq["rmaxis"], eq["zmaxis"])
    psi_top = 1.0 - 2.0 * closure["w_ped"]
    lcfs = closed_surface(eq, closure["psi_lcfs"])
    top = closed_surface(eq, psi_top)
    seg = np.diff(lcfs, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    r_mid_seg = 0.5 * (lcfs[1:, 0] + lcfs[:-1, 0])
    area = float(np.sum(2.0 * np.pi * r_mid_seg * seg_len))
    r_out = float(by_angle(lcfs, axis, 1)[0, 0])
    r_top = float(by_angle(top, axis, 1)[0, 0])
    out = {"S_lcfs": area, "perimeter": float(np.sum(seg_len)), "R_out_mid": r_out,
           "delta_ped": max(r_out - r_top, 1e-3), "psi_top": psi_top,
           "rho_top": rho_of_psi(eq, psi_top)}
    out.update(divertor_geometry(eq, fields, r_out, closure))
    return out


# ---- closures ------------------------------------------------------------------------

def seed_pressure(spec, closure=CLOSURE):
    beta_t = closure["beta_N_seed"] / 100.0 * (spec["Ip"] / 1e6) / (spec["a"] * spec["B0"])
    return 3.0 * beta_t * spec["B0"] ** 2 / (2.0 * MU0)


def edge_pressure_L(p_sol, n_ped, t_sep, delta_ped, s_lcfs, press_factor, closure=CLOSURE):
    t_ped = t_sep + p_sol * delta_ped / (press_factor * n_ped * E_CHARGE * closure["chi_L"]
                                         * s_lcfs)
    return E_CHARGE * n_ped * t_ped * press_factor


def barrier_pressure(b_pol_avg, closure=CLOSURE):
    return closure["beta_p_ped_H"] * b_pol_avg ** 2 / (2.0 * MU0)


def load_boundary(path):
    with open(path) as handle:
        raw = json.load(handle)
    return {key: np.array([np.nan if v is None else v for v in value], dtype=float)
            for key, value in raw.items()}


def overlap_ratio(boundary, psi_min):
    first = boundary.get("alpha_critical", boundary.get("alpha_critical1"))
    mask = ((boundary["psi"] >= psi_min) & np.isfinite(boundary["alpha"]) & np.isfinite(first)
            & (first > 0))
    if not mask.any():
        return float("nan")
    return float(np.max(boundary["alpha"][mask] / first[mask]))


def second_stable_access(boundary, psi_min):
    first = boundary.get("alpha_critical1", boundary.get("alpha_critical"))
    second = boundary.get("alpha_critical2", np.full_like(first, np.nan))
    mask = (boundary["psi"] >= psi_min) & np.isfinite(boundary["alpha"])
    alpha = boundary["alpha"][mask]
    a1, a2 = first[mask], second[mask]
    above_first = np.isfinite(a1) & (alpha > a1)
    above_second = np.isfinite(a2) & (alpha > a2)
    return not bool(np.any(above_first & ~above_second))


def second_stable(boundary, psi_min):
    first = boundary.get("alpha_critical1", boundary.get("alpha_critical"))
    second = boundary.get("alpha_critical2")
    if second is None:
        return False
    mask = (boundary["psi"] >= psi_min) & np.isfinite(boundary["alpha"])
    alpha, a1, a2 = boundary["alpha"][mask], first[mask], second[mask]
    above_first = np.isfinite(a1) & (alpha > a1)
    if not above_first.any():
        return False
    above_second = np.isfinite(a2) & (alpha > a2)
    return bool(np.all(above_second[above_first]))


def band_width(boundary, psi_min):
    first = boundary.get("alpha_critical1", boundary.get("alpha_critical"))
    second = boundary.get("alpha_critical2")
    if second is None:
        return float("nan")
    mask = (boundary["psi"] >= psi_min) & np.isfinite(first) & (first > 0)
    if mask.sum() < 2:
        return float("nan")
    a1, a2 = first[mask], second[mask]
    ratio = np.where(np.isfinite(a2), a2 / a1, np.inf)
    return float(np.median(ratio))


SCAN_RATIO_MAX = 1.7


def band_at_first_boundary(scan):
    points = sorted((float(x["ratio"]), 0.0 if np.isinf(x["band"]) else 1.0 / float(x["band"]))
                    for x in scan if x.get("solved") and x.get("ratio") is not None
                    and x["ratio"] <= SCAN_RATIO_MAX
                    and x.get("band") is not None and not np.isnan(x["band"]))
    if not points:
        return float("nan")
    below = [point for point in points if point[0] <= 1.0]
    above = [point for point in points if point[0] > 1.0]
    if below and above:
        (ratio_0, close_0), (ratio_1, close_1) = below[-1], above[0]
        closeness = close_0 + (close_1 - close_0) * (1.0 - ratio_0) / (ratio_1 - ratio_0)
    elif below:
        closeness = min(close for _, close in below)
    else:
        closeness = above[0][1]
    return float("inf") if closeness <= 0.0 else float(1.0 / closeness)


def access_of(scan, closure=CLOSURE):
    band = band_at_first_boundary(scan)
    return bool(not np.isnan(band) and band < closure["band_max"])


def polygon_area(line):
    x, y = line[:, 0], line[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def check_nested(eq, levels=None):
    levels = np.linspace(0.05, 0.98, 20) if levels is None else levels
    axis = (eq["rmaxis"], eq["zmaxis"])
    problems, previous = [], 0.0
    for level in levels:
        closed = [line for line in contour_lines(eq, float(level))
                  if len(line) >= 8 and np.hypot(*(line[0] - line[-1])) <= 0.05
                  and inside_polygon(axis, line)]
        if not closed:
            problems.append(f"psi_n {level:.2f}: no closed surface around the axis")
            continue
        if len(closed) > 1:
            problems.append(f"psi_n {level:.2f}: {len(closed)} closed surfaces around the axis")
        area = min(polygon_area(line) for line in closed)
        if area <= previous:
            problems.append(f"psi_n {level:.2f}: surface area {area:.4f} below the one inside")
        previous = area
    return problems


def marginal_pedestal(pressures, ratios):
    p = np.asarray(pressures, dtype=float)
    r = np.asarray(ratios, dtype=float)
    valid = np.isfinite(r) & (r > 0) & (p > 0)
    if valid.sum() == 0:
        return float("nan")
    p, r = p[valid], r[valid]
    order = np.argsort(p)
    p, r = p[order], r[order]
    if p.size == 1:
        return float(p[0] / r[0])
    unstable = np.flatnonzero(r >= 1.0)
    if unstable.size == 0:
        i0, i1 = p.size - 2, p.size - 1
    elif unstable[0] == 0:
        i0, i1 = 0, 1
    else:
        i0, i1 = unstable[0] - 1, unstable[0]
    lp, lr = np.log(p[[i0, i1]]), np.log(r[[i0, i1]])
    if abs(lr[1] - lr[0]) < 1e-9:
        return float(p[i1] / r[i1])
    slope = (lp[1] - lp[0]) / (lr[1] - lr[0])
    marginal = float(np.exp(lp[0] - lr[0] * slope))
    return float(np.clip(marginal, 0.05 * p[0], 3.0 * p[-1]))


def lambda_q_mm(barrier, b_pol_mid, p_ped, p_ped_bal, l_par, t_sep, closure=CLOSURE):
    c0, e0 = closure["lambda_eich"]
    drift = c0 * b_pol_mid ** e0
    s_l = 0.0 if barrier else float(np.clip(1.0 - p_ped / max(p_ped_bal, 1e-9), 0.0, 1.0))
    c_s = np.sqrt(2.0 * E_CHARGE * t_sep / (closure["A_i"] * M_PROTON))
    turbulent = 1e3 * np.sqrt(closure["chi_L"] * s_l * l_par / c_s)
    return float(drift + turbulent), float(drift), float(turbulent)


def peak_heat_flux(p_sol_w, r_strike, lambda_q_mm_value, f_exp, sin_target=1.0,
                    closure=CLOSURE):
    return float(closure["f_outer"] * p_sol_w / 1e6 * sin_target
                 / (2.0 * np.pi * r_strike * lambda_q_mm_value * 1e-3 * f_exp))


def impurity_ratio(barrier, p_ped, p_ped_l, closure=CLOSURE):
    c0, c1 = closure["c_imp"]
    return float(c0 + c1 * (1.0 if barrier else 0.0) * max(p_ped / max(p_ped_l, 1e-9) - 1.0, 0.0))


# ---- TORAX ---------------------------------------------------------------------------

def density_profile(spec, closure=CLOSURE):
    n_ped = spec["n_e_ped"]
    psi = np.linspace(0.0, 1.0, N_PSI)
    return psi, pedestal_profile(closure["n_sep_over_ped"] * n_ped, n_ped,
                                 closure["n_core_over_ped"] * n_ped, psi, width=closure["w_ped"])


def torax_config(spec, geqdsk_name, geometry_dir, p_ped, rho_top, rho, n_e_rho, closure=CLOSURE,
                 t_final=None):
    t_sep_kev = spec["T_sep"] / 1e3
    n_sep = closure["n_sep_over_ped"] * spec["n_e_ped"]
    n_profile = {float(round(r, 4)): float(v) for r, v in zip(rho, n_e_rho)}
    return {
        "plasma_composition": {"main_ion": "D", "impurity": spec["impurity"],
                               "Z_eff": spec["Z_eff"]},
        "profile_conditions": {
            "Ip": spec["Ip"],
            "T_i": {0.0: {0.0: 2.0, 1.0: t_sep_kev}}, "T_i_right_bc": t_sep_kev,
            "T_e": {0.0: {0.0: 2.0, 1.0: t_sep_kev}}, "T_e_right_bc": t_sep_kev,
            "n_e": {0.0: n_profile}, "n_e_right_bc": n_sep,
            "normalize_n_e_to_nbar": False, "n_e_nbar_is_fGW": False,
        },
        "numerics": {"t_final": float(t_final or TORAX["t_final"]), "evolve_ion_heat": True,
                     "evolve_electron_heat": True, "evolve_density": False,
                     "evolve_current": False, "max_dt": TORAX["max_dt"],
                     "chi_timestep_prefactor": 50, "dt_reduction_factor": 3},
        "geometry": {"geometry_type": "eqdsk", "geometry_file": geqdsk_name,
                     "geometry_directory": str(geometry_dir), "cocos": TORAX["cocos"],
                     "Ip_from_parameters": True, "n_surfaces": TORAX["n_surfaces"],
                     "last_surface_factor": TORAX["last_surface_factor"],
                     "n_rho": TORAX["n_rho"]},
        "sources": {"ei_exchange": {}, "ohmic": {}, "bremsstrahlung": {},
                    "cyclotron_radiation": {},
                    "impurity_radiation": {"model_name": "mavrin_fit"},
                    "generic_heat": {"gaussian_location": spec["heating_location"],
                                     "gaussian_width": spec["heating_width"],
                                     "P_total": spec["P_heat"],
                                     "electron_heat_fraction": spec["electron_heat_fraction"]}},
        "pedestal": {"model_name": "set_P_ped_n_ped", "set_pedestal": True, "P_ped": float(p_ped),
                     "n_e_ped": spec["n_e_ped"], "T_i_T_e_ratio": 1.0,
                     "rho_norm_ped_top": float(rho_top)},
        "transport": {"model_name": TORAX["transport"], "chi_min": TORAX["chi_min"],
                      "chi_max": TORAX["chi_max"], "D_e_min": TORAX["D_e_min"]},
        "solver": {"solver_type": "linear", "use_predictor_corrector": True,
                   "n_corrector_steps": 10, "use_pereverzev": True, "chi_pereverzev": 30,
                   "D_pereverzev": 15},
        "time_step_calculator": {"calculator_type": "chi"},
    }


def to_rho_grid(values, rho, rho_face, rho_cell):
    values = np.asarray(values, dtype=float)
    for grid in (rho, rho_cell, rho_face):
        if grid is not None and values.size == grid.size:
            return values if grid is rho else np.interp(rho, grid, values)
    raise ValueError(f"profile of length {values.size} matches no TORAX grid")


PROFILE_NAMES = ("T_e", "T_i", "n_e", "n_i", "pressure_thermal_total", "q", "psi", "volume",
                 "vpr", "area", "g0", "g1", "Z_eff", "chi_turb_i", "chi_turb_e")
SCALAR_NAMES = ("P_SOL_total", "P_LH", "P_radiation_e", "P_external_total", "P_heat_total",
                "P_aux_total", "P_ohmic_e", "W_thermal_total", "tau_E", "n_e_line_avg",
                "n_e_volume_avg", "T_e_volume_avg", "T_i_volume_avg", "Ip", "B_0", "R_major",
                "a_minor", "A_i", "H98", "q95", "beta_N", "dW_thermal_dt_smoothed")


def read_run(nc_path):
    import xarray as xr

    tree = xr.open_datatree(nc_path)
    profiles = tree["profiles"].to_dataset().isel(time=-1)
    scalars = tree["scalars"].to_dataset().isel(time=-1)
    coords = {}
    for name in ("rho_norm", "rho_face_norm", "rho_cell_norm"):
        for node in (profiles, tree):
            if name in node.coords:
                coords[name] = np.asarray(node[name].values, dtype=float)
                break
    rho, rho_face, rho_cell = coords["rho_norm"], coords.get("rho_face_norm"), coords.get("rho_cell_norm")
    out = {"rho": rho, "profiles": {}, "scalars": {}}
    for name in PROFILE_NAMES:
        if name in profiles:
            out["profiles"][name] = to_rho_grid(profiles[name].values, rho, rho_face, rho_cell)
    for name in SCALAR_NAMES:
        if name in scalars:
            out["scalars"][name] = float(np.asarray(scalars[name].values))
    numerics = tree["numerics"].to_dataset() if "numerics" in tree.children else None
    code = (int(np.asarray(numerics["sim_error"].values).ravel()[-1])
            if numerics is not None and "sim_error" in numerics else -1)
    out["sim_error_code"] = code
    out["sim_error"] = "SimError.NO_ERROR" if code == 0 else f"SimError({code})"
    psi = out["profiles"]["psi"]
    out["psi_n"] = (psi - psi[0]) / (psi[-1] - psi[0])
    tree.close()
    return out


def run_torax(torax, config, name, out_dir):
    started = time.time()
    nc_path = Path(out_dir) / f"{name}.nc"
    key_path = Path(out_dir) / f"{name}.nc.key"
    key = hashlib.sha256(json.dumps(config, sort_keys=True, default=jsonable).encode()).hexdigest()
    if nc_path.is_file() and key_path.is_file() and key_path.read_text().strip() == key:
        run = read_run(nc_path)
        run["seconds"], run["cached"] = 0.0, True
        return run
    for stale in (nc_path, key_path):
        stale.unlink(missing_ok=True)
    summary = torax.simulate(config, output_name=name, out_dir=out_dir)
    key_path.write_text(key)
    run = read_run(nc_path)
    run["seconds"] = round(time.time() - started, 1)
    run["torax_summary"] = summary
    run["sim_error"] = str(summary.get("sim_error", run["sim_error"]))
    return run


def solve_equilibrium(equil, config, name, out_dir, psi_init=None, log=None, closure=CLOSURE):
    geqdsk, summary = Path(out_dir) / f"{name}.geqdsk", Path(out_dir) / f"{name}.json"
    say = log or (lambda *args: None)
    if geqdsk.is_file() and summary.is_file():
        with open(summary) as handle:
            stored = json.load(handle)
        asked = stored.get("config") or {}
        if all(np.isclose(float(asked.get(key, np.nan)), float(config[key]), rtol=1e-6)
               for key in ("p_axis", "Ip")):
            return {"stats": stored.get("stats", {}), "diverted": stored.get("diverted"),
                    "coil_currents": stored.get("coil_currents"), "cached": True}, None

    def attempt(cfg, psi):
        try:
            return equil.equilibrium(cfg, name, psi_init=psi, out_dir=out_dir)
        except ValueError as first:
            try:
                return equil.equilibrium(dict(cfg, maxits=RETRY_MAXITS, urf=RETRY_URF), name,
                                         psi_init=None, out_dir=out_dir)
            except ValueError as second:
                raise ValueError(f"{first}; retried from scratch, relaxed: {second}") from second

    result = attempt(config, psi_init)
    problems = check_nested(read_geqdsk(geqdsk))
    if problems:
        # the coils were let go too far: pull them harder to the reference and solve again
        say(f"  [{name}] surfaces not nested ({problems[0]}); solving again with the coils "
            f"held {closure['coil_weight_fallback']:g} times harder")
        stronger = dict(config, coil_weight=float(config.get("coil_weight", closure["coil_weight"]))
                        * closure["coil_weight_fallback"])
        result = attempt(stronger, None)
        problems = check_nested(read_geqdsk(geqdsk))
        if problems:
            # the files of a failed solve are removed, or a later run would take them as done
            for path in (geqdsk, summary):
                path.unlink(missing_ok=True)
            raise ValueError(f"{name}: surfaces still not nested: {problems[:3]}")
        result[0]["coil_weight"] = stronger["coil_weight"]
    return result


def evaluate_ballooning(stability, geqdsk_names, both, names, out_dir):
    # every equilibrium without a result yet goes to GPEC in one call
    missing = [(g, n) for g, n in zip(geqdsk_names, names)
               if not (Path(out_dir) / f"{n}.json").is_file()]
    if missing:
        stability.ballooning_many([g for g, _ in missing], both=both,
                                  output_names=[n for _, n in missing], out_dir=out_dir)
    return [load_boundary(Path(out_dir) / f"{n}.json") for n in names]


def alpha_profile(run, spec):
    p = run["profiles"]["pressure_thermal_total"]
    q = run["profiles"]["q"]
    rho = run["rho"]
    r_major = run["scalars"].get("R_major", spec["R0"])
    a = run["scalars"].get("a_minor", spec["a"])
    b0 = run["scalars"].get("B_0", spec["B0"])
    dp_dr = np.gradient(p, rho) / a
    return -2.0 * MU0 * r_major * q ** 2 * dp_dr / b0 ** 2


# ---- the coupled model ------------------------------------------------------------------

def equilibrium_config(spec, p_axis, kinetic=None, closure=CLOSURE):
    config = {"R0": spec["R0"], "Z0": spec["Z0"], "a": spec["a"], "kappa": spec["kappa"],
              "delta": spec["delta"], "kappa_lower": spec["kappa_lower"],
              "delta_lower": spec["delta_lower"], "x_point": "lower",
              "machine_scale": spec["machine_scale"], "Ip": spec["Ip"], "B0": spec["B0"],
              "p_axis": float(p_axis), "coil_weight": closure["coil_weight"],
              "coil_bound_factor": closure["coil_bound_factor"],
              "isoflux_weight": closure["isoflux_weight"]}
    if kinetic is not None:
        config["kinetic"] = kinetic
    return config


def core_on_psi(run, psi):
    # the temperatures of a TORAX run, in eV, as functions of normalised poloidal flux
    out = {}
    for key in ("T_e", "T_i"):
        out[key] = np.interp(psi, run["psi_n"], run["profiles"][key] * 1e3)
    return out


def scan_kinetic(spec, core, psi, t_ped_ev, closure=CLOSURE):
    psi_top = 1.0 - 2.0 * closure["w_ped"]
    _, n_e = density_profile(spec, closure)
    kinetic = {"n_e": n_e, "Z_eff": spec["Z_eff"], "Z_imp": closure["Z_imp"]}
    for key in ("T_e", "T_i"):
        t_core = float(np.interp(0.0, psi, core[key]) - np.interp(psi_top, psi, core[key]) + t_ped_ev)
        t_core = max(t_core, 1.5 * t_ped_ev)
        kinetic[key] = pedestal_profile(spec["T_sep"], t_ped_ev, t_core, psi, width=closure["w_ped"])
    return kinetic


def kinetic_axis_pressure(kinetic, press_factor):
    return float(E_CHARGE * kinetic["n_e"][0] * kinetic["T_e"][0] * press_factor)


def seed_and_first_run(spec, sim, name, work, say, closure, timing, psi_grid, n_e_psi, n_ped,
                       press_factor, rho_a):
    t0 = time.time()
    eq0, psi_prev = solve_equilibrium(sim.equil,
                                      equilibrium_config(spec, seed_pressure(spec, closure)),
                                      f"{name}_eq0", work, log=say, closure=closure)
    g0 = read_geqdsk(work / f"{name}_eq0.geqdsk")
    geo0 = geometry_of(g0, closure)
    timing["equilibrium_seed"] = round(time.time() - t0, 1)
    say(f"  [{name}] seed equilibrium {timing['equilibrium_seed']}s, q95 "
        f"{eq0['stats'].get('q_95')}, diverted {eq0['diverted']}")
    if eq0["diverted"] is False:
        raise ValueError(f"the shape of {name} is not diverted: its X-point sits outside the "
                         f"plasma; move it away from the wall")
    shape_seed = shape_of(g0, closure)
    say(f"  [{name}] shape asked {shape_text(spec)}")
    say(f"  [{name}] shape got   {shape_text(shape_seed)} (error {shape_error(spec, shape_seed)})")

    # 2. first TORAX run, barrier-free edge from the power guess
    t0 = time.time()
    n_e_rho = np.interp([psi_of_rho(g0, r) for r in rho_a], psi_grid, n_e_psi)
    p_ped_l0 = edge_pressure_L(closure["P_SOL_guess"] * spec["P_heat"], n_ped, spec["T_sep"],
                               geo0["delta_ped"], geo0["S_lcfs"], press_factor, closure)
    # the first run only supplies the boundary power and the core profiles, so it is short
    run_a = run_torax(sim.torax, torax_config(spec, f"{name}_eq0.geqdsk", work, p_ped_l0,
                                              geo0["rho_top"], rho_a, n_e_rho, closure,
                                              t_final=TORAX["t_final_first"]),
                      f"{name}_runA", work)
    p_sol, p_lh = run_a["scalars"]["P_SOL_total"], run_a["scalars"]["P_LH"]
    timing["torax_first"] = run_a["seconds"]
    say(f"  [{name}] first TORAX run {run_a['seconds']}s, P_SOL {p_sol / 1e6:.2f} MW, "
        f"P_LH {p_lh / 1e6:.2f} MW, {run_a['sim_error']}")
    return eq0, g0, geo0, shape_seed, run_a, psi_prev


def prepare_edge(spec, sim, name, work, say, closure, timing, psi_grid, n_e_psi, n_ped,
                 press_factor, rho_a):
    eq0, g0, geo0, shape_seed, run_a, psi_prev = seed_and_first_run(
        spec, sim, name, work, say, closure, timing, psi_grid, n_e_psi, n_ped, press_factor,
        rho_a)
    props = build_props(run_a, g0, spec, closure)
    p_sol, p_lh = run_a["scalars"]["P_SOL_total"], run_a["scalars"]["P_LH"]

    # 3. the pedestal of the barrier-free edge
    b_pol_avg = MU0 * spec["Ip"] / geo0["perimeter"]
    p_ped_l = edge_pressure_L(p_sol, n_ped, spec["T_sep"], geo0["delta_ped"], geo0["S_lcfs"],
                              press_factor, closure)
    core = core_on_psi(run_a, psi_grid)

    t0 = time.time()
    p_ped_a = min(p_ped_l, closure["access_beta_p"] * b_pol_avg ** 2 / (2.0 * MU0))
    points = [("A", p_ped_a)]
    if p_ped_l > 1.05 * p_ped_a:
        points.append(("L", p_ped_l))
    pressures, ratios, bands, scan, solved = [], [], [], [], []
    for k, (label, p_point) in enumerate(points):
        tries = [1.0] + (list(NUDGE_FACTORS) if label == "A" else [])
        eq_k = None
        for factor_k in tries:
            p_ped_k = factor_k * p_point
            t_ped = p_ped_k / (E_CHARGE * n_ped * press_factor)
            kinetic = scan_kinetic(spec, core, psi_grid, t_ped, closure)
            config = equilibrium_config(spec, kinetic_axis_pressure(kinetic, press_factor),
                                        kinetic, closure=closure)
            try:
                eq_k, psi_prev = solve_equilibrium(sim.equil, config, f"{name}_scan{k}", work,
                                                   psi_init=psi_prev, log=say, closure=closure)
                break
            except ValueError as exc:
                say(f"  [{name}] scan {label}: p_ped {p_ped_k / 1e3:.2f} kPa has no equilibrium "
                    f"({exc})")
        if eq_k is None:
            scan.append({"point": label, "p_ped": p_point, "factor": 1.0, "ratio": None,
                         "solved": False})
            continue
        if factor_k != 1.0:
            say(f"  [{name}] scan {label}: taken at {factor_k:g} of its pedestal, "
                f"{p_ped_k / 1e3:.2f} kPa")
        solved.append((k, label, p_ped_k, t_ped, eq_k, factor_k))
    if not solved or solved[0][1] != "A":
        raise RuntimeError(f"the barrier-free edge of {name} has no equilibrium at the pedestal "
                           f"the band is read on, {p_ped_a / 1e3:.2f} kPa")
    # both boundaries on every pedestal of the scan, in one GPEC call
    t1 = time.time()
    boundaries = evaluate_ballooning(sim.stability, [f"{name}_scan{k}.geqdsk" for k, *_ in solved],
                                     True, [f"{name}_bal{k}" for k, *_ in solved], work)
    gpec_seconds = round(time.time() - t1, 1)
    for (k, label, p_ped_k, t_ped, eq_k, factor_k), boundary in zip(solved, boundaries):
        ratio = overlap_ratio(boundary, closure["psi_pedestal"])
        band = band_width(boundary, closure["psi_band"])
        stable = second_stable_access(boundary, closure["psi_pedestal"])
        pressures.append(p_ped_k)
        ratios.append(ratio)
        bands.append(band)
        scan.append({"point": label, "p_ped": p_ped_k, "factor": factor_k, "T_ped_eV": t_ped,
                     "ratio": ratio, "band": band, "solved": True, "stable": stable,
                     "gpec_seconds": gpec_seconds,
                     "q_95": eq_k["stats"].get("q_95"),
                     "shape": shape_text(shape_of(read_geqdsk(work / f"{name}_scan{k}.geqdsk"),
                                                  closure))})
        say(f"  [{name}] scan {label}: p_ped {p_ped_k / 1e3:.2f} kPa, max alpha/alpha_crit "
            f"{ratio:.3f}, band {band:.2f}, stable {stable}")
    say(f"  [{name}] GPEC on {len(solved)} equilibria: {gpec_seconds}s")
    p_ped_bal = marginal_pedestal(pressures, ratios)
    if not np.isfinite(p_ped_bal):
        # no first boundary within reach of any pedestal scanned: nothing clamps the edge
        p_ped_bal = float("inf")
    scan.sort(key=lambda entry: entry["p_ped"])
    band_edge = bands[0]
    if np.isnan(band_edge):
        raise RuntimeError(f"GPEC found no first boundary on the steep pedestal surfaces of "
                           f"the barrier-free edge of {name}; the band cannot be read")
    timing["scan"] = round(time.time() - t0, 1)

    stage = {"P_SOL": p_sol, "P_LH": p_lh, "B_pol_avg": b_pol_avg, "p_ped_L": p_ped_l,
             "p_ped_A": p_ped_a,
             "core": {key: np.asarray(value).tolist() for key, value in core.items()},
             "p_ped_ballooning": p_ped_bal, "band_edge": band_edge, "scan": scan,
             "geometry_seed": geo0, "shape_seed": shape_seed,
             "coil_currents_seed": eq0.get("coil_currents"),
             "q_95_seed": eq0["stats"].get("q_95"), "sim_error_first": run_a["sim_error"],
             "timing": {key: timing[key] for key in ("equilibrium_seed", "torax_first", "scan")},
             "props": props}
    return {"eq0": eq0, "geo0": geo0, "shape_seed": shape_seed,
            "sim_error_first": run_a["sim_error"], "P_SOL": p_sol, "P_LH": p_lh,
            "B_pol_avg": b_pol_avg, "p_ped_L": p_ped_l, "core": core, "psi_prev": psi_prev,
            "solved": solved, "scan": scan, "p_ped_ballooning": p_ped_bal,
            "band_edge": band_edge, "props": props, "stage": stage}


def restore_edge(stage, timing, say, name):
    timing.update(stage.get("timing", {}))
    say(f"  [{name}] seed equilibrium, first TORAX run and scan taken from the stage record")
    props = stage.get("props")
    if props is not None:
        props = {key: (np.asarray(value) if np.ndim(value) else float(value))
                 for key, value in props.items()}
    return {"eq0": {"coil_currents": stage.get("coil_currents_seed"),
                    "stats": {"q_95": stage.get("q_95_seed")}},
            "geo0": stage["geometry_seed"], "shape_seed": stage["shape_seed"],
            "sim_error_first": stage["sim_error_first"],
            "P_SOL": float(stage["P_SOL"]), "P_LH": float(stage["P_LH"]),
            "B_pol_avg": float(stage["B_pol_avg"]), "p_ped_L": float(stage["p_ped_L"]),
            "core": {key: np.asarray(value, dtype=float) for key, value in stage["core"].items()},
            "psi_prev": None, "solved": [], "scan": list(stage["scan"]),
            "p_ped_ballooning": float(stage["p_ped_ballooning"]),
            "band_edge": float(stage["band_edge"]), "props": props, "stage": stage}


def run_setting(spec, sim, name, workdir, reveal=False, log=None, closure=CLOSURE, stage=None,
                on_stage=None):
    say = log or (lambda *args: None)
    spec = complete_spec(spec)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    timing, started = {}, time.time()
    z_imp = IMPURITY_CHARGE[spec["impurity"]]
    n_i_over_n_e, imp_frac, press_factor = composition(spec["Z_eff"], z_imp)
    n_ped = spec["n_e_ped"]
    psi_grid, n_e_psi = density_profile(spec, closure)

    rho_a = np.linspace(0.0, 1.0, 41)
    if stage is None:
        edge = prepare_edge(spec, sim, name, work, say, closure, timing, psi_grid, n_e_psi,
                            n_ped, press_factor, rho_a)
        if on_stage is not None:
            on_stage(edge["stage"])
    else:
        edge = restore_edge(stage, timing, say, name)
        if edge["props"] is None:
            eq0, g0, _, _, run_a, _ = seed_and_first_run(
                spec, sim, name, work, say, closure, {}, psi_grid, n_e_psi, n_ped,
                press_factor, rho_a)
            edge["props"] = build_props(run_a, g0, spec, closure)
            edge["eq0"] = eq0
            edge["stage"]["props"] = edge["props"]
            if on_stage is not None:
                on_stage(edge["stage"])
    eq0, geo0, shape_seed = edge["eq0"], edge["geo0"], edge["shape_seed"]
    sim_error_first = edge["sim_error_first"]
    p_sol, p_lh, b_pol_avg = edge["P_SOL"], edge["P_LH"], edge["B_pol_avg"]
    p_ped_l, core = edge["p_ped_L"], edge["core"]
    psi_prev, solved, scan = edge["psi_prev"], edge["solved"], edge["scan"]
    p_ped_bal, band_edge = edge["p_ped_ballooning"], edge["band_edge"]
    # the barrier never holds less than the barrier-free edge does
    p_ped_h = max(barrier_pressure(b_pol_avg, closure), p_ped_l)
    band_first = band_at_first_boundary(scan)
    access = access_of(scan, closure)

    # 5. the edge closure
    barrier = access and p_sol >= p_lh
    p_ped_target = p_ped_h if barrier else float(min(p_ped_bal, p_ped_l))
    say(f"  [{name}] band {band_first:.2f} at the first boundary, {band_edge:.2f} on the scanned "
        f"pedestal (limit {closure['band_max']:g}) -> access {access}, "
        f"P_SOL/P_LH {p_sol / max(p_lh, 1.0):.2f} -> barrier {barrier}; p_ped "
        f"{p_ped_target / 1e3:.2f} kPa (H {p_ped_h / 1e3:.2f}, L {p_ped_l / 1e3:.2f}, "
        f"ballooning {p_ped_bal / 1e3:.2f})")

    steps = (1.0,) + tuple(closure["barrier_ladder"] if barrier else NUDGE_FACTORS)
    if barrier:
        steps += tuple(f for f in BARRIER_LADDER_LOW if f * p_ped_target >= p_ped_l)
    final = work / f"{name}_eq.geqdsk"
    p_ped, p_ped_step, eq_f, run_b = None, None, None, None
    timing["equilibrium_final"], timing["torax_final"] = 0.0, 0.0
    for step in steps:
        t0 = time.time()
        p_try = step * p_ped_target
        same = [k for k, _, p_ped_k, *_ in solved if abs(p_ped_k - p_try) <= 1e-3 * p_try]
        if same and not final.is_file():
            for suffix in (".geqdsk", ".json"):
                shutil.copy2(work / f"{name}_scan{same[0]}{suffix}", work / f"{name}_eq{suffix}")
        t_ped = p_try / (E_CHARGE * n_ped * press_factor)
        kinetic = scan_kinetic(spec, core, psi_grid, t_ped, closure)
        config = equilibrium_config(spec, kinetic_axis_pressure(kinetic, press_factor), kinetic,
                                    closure=closure)
        try:
            eq_try, _ = solve_equilibrium(sim.equil, config, f"{name}_eq", work,
                                          psi_init=psi_prev, log=say, closure=closure)
        except ValueError as exc:
            say(f"  [{name}] final equilibrium at {p_try / 1e3:.2f} kPa failed ({exc})")
            timing["equilibrium_final"] += round(time.time() - t0, 1)
            continue
        g_try = read_geqdsk(final)
        geo_try = geometry_of(g_try, closure)
        timing["equilibrium_final"] += round(time.time() - t0, 1)
        n_e_rho = np.interp([psi_of_rho(g_try, r) for r in rho_a], psi_grid, n_e_psi)
        run_try = run_torax(sim.torax, torax_config(spec, f"{name}_eq.geqdsk", work, p_try,
                                                    geo_try["rho_top"], rho_a, n_e_rho, closure),
                            f"{name}_run", work)
        timing["torax_final"] += run_try["seconds"]
        if not str(run_try["sim_error"]).endswith("NO_ERROR"):
            say(f"  [{name}] TORAX inside the equilibrium at {p_try / 1e3:.2f} kPa ended with "
                f"{run_try['sim_error']}")
            continue
        p_ped, p_ped_step = float(p_try), float(step)
        eq_f, g_f, geo_f, run_b = eq_try, g_try, geo_try, run_try
        break
    if run_b is None:
        raise RuntimeError(f"no equilibrium and TORAX run of {name} hold the pedestal of the edge "
                           f"closure, {p_ped_target / 1e3:.2f} kPa, nor any of {steps[1:]} of it")
    if p_ped_step != 1.0:
        say(f"  [{name}] the pedestal is held at {p_ped_step:g} of its closure value")
    shape_final = shape_of(g_f, closure)
    say(f"  [{name}] final shape {shape_text(shape_final)} (error {shape_error(spec, shape_final)})")
    p_sol_b = run_b["scalars"]["P_SOL_total"]

    lam, lam_drift, lam_turb = lambda_q_mm(barrier, geo0["B_pol_mid"], p_ped, p_ped_bal,
                                           geo0["L_par"], spec["T_sep"], closure)
    alpha = alpha_profile(run_b, spec)
    truth = {"barrier": 1.0 if barrier else 0.0,
             "alpha_edge": float(np.interp(RHO_ALPHA, run_b["rho"], alpha)),
             "lambda_q": lam,
             "q_peak": peak_heat_flux(p_sol_b, geo0["R_strike"], lam, geo0["f_exp"],
                                      geo0["sin_target"], closure),
             "tau_imp": impurity_ratio(barrier, p_ped, p_ped_l, closure)}
    props = edge["props"]
    timing["total"] = round(time.time() - started, 1)
    summary = {"setting": spec, "truth": truth, "props": props,
               "profiles": {key: run_b["profiles"][key] for key in
                            ("T_e", "T_i", "n_e", "pressure_thermal_total", "q")
                            if key in run_b["profiles"]},
               "alpha": alpha, "scalars": run_b["scalars"], "sim_error": run_b["sim_error"],
               "files": {"equilibrium": f"{name}_eq.geqdsk", "run": f"{name}_run.nc"},
               "timing": timing}
    closure_record = {"access": access, "band_edge": band_edge, "band_first": band_first,
                      "band_max": closure["band_max"],
                      "p_ped_target": p_ped_target, "p_ped_step": p_ped_step,
                      "coil_currents_seed": eq0.get("coil_currents"),
                      "coil_currents_final": eq_f.get("coil_currents"),
                      "shape_asked": shape_text(spec), "shape_seed": shape_text(shape_seed),
                      "shape_final": shape_text(shape_final),
                      "shape_error_final": shape_error(spec, shape_final),
                      "p_ped_run": float(np.interp(geo_f["rho_top"], run_b["rho"],
                                                   run_b["profiles"]["pressure_thermal_total"])),
                      "B_pol_avg": b_pol_avg,
                      "P_SOL_first": p_sol, "P_LH_first": p_lh,
                      "P_SOL": p_sol_b, "P_LH": run_b["scalars"].get("P_LH"),
                      "p_ped": p_ped, "p_ped_H": p_ped_h, "p_ped_L": p_ped_l,
                      "p_ped_A": edge["stage"].get("p_ped_A"),
                      "p_ped_ballooning": p_ped_bal, "scan": scan,
                      "lambda_q_drift": lam_drift, "lambda_q_turbulent": lam_turb,
                      "geometry_seed": geo0, "geometry_final": geo_f,
                      "sim_error_first": sim_error_first, "tau_E": run_b["scalars"].get("tau_E")}
    if reveal:
        summary["closure"] = closure_record
    with open(work / f"{name}.json", "w") as handle:
        json.dump(summary, handle, default=jsonable)
    return summary


def shape_text(source):
    # the shape moments of a setting or of an equilibrium's statistics, in one line
    keys = (("R0", "R_geo"), ("a", "a_geo"), ("kappa", "kappaU"), ("kappa_lower", "kappaL"),
            ("delta", "deltaU"), ("delta_lower", "deltaL"))
    parts = []
    for spec_key, stats_key in keys:
        value = source.get(spec_key, source.get(stats_key))
        if value is not None:
            parts.append(f"{stats_key} {float(value):.3f}")
    return " ".join(parts)


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def resample_closed(points, n):
    # a closed polyline at n points equally spaced along it, starting from its first point
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    ring = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(ring, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    wanted = np.linspace(0.0, along[-1], n, endpoint=False)
    return np.column_stack([np.interp(wanted, along, ring[:, 0]),
                            np.interp(wanted, along, ring[:, 1])])


def build_props(run, eq, spec, closure=CLOSURE):
    rho = run["rho"]
    axis = (eq["rmaxis"], eq["zmaxis"])
    fields = Fields(eq)
    k = K_THETA
    r_pts = np.zeros((rho.size, k))
    z_pts = np.zeros((rho.size, k))
    b_tot = np.zeros((rho.size, k))
    b_pol = np.zeros((rho.size, k))
    for i, psi_n in enumerate(run["psi_n"]):
        level = float(min(psi_n, closure["psi_lcfs"]))
        if level <= 1e-3:
            pts = np.tile(np.asarray(axis, dtype=float), (k, 1))
        else:
            pts = by_angle(closed_surface(eq, level), axis, k)
        bp, _, bt, _ = fields(pts)
        r_pts[i], z_pts[i], b_tot[i], b_pol[i] = pts[:, 0], pts[:, 1], bt, bp
    prof = run["profiles"]
    vpr = prof["vpr"].copy()
    safe_vpr = np.where(np.abs(vpr) > 1e-12, vpr, np.nan)
    grad_rho = np.nan_to_num(prof["g0"] / safe_vpr, nan=0.0)
    grad_rho_sq = np.nan_to_num(prof["g1"] / safe_vpr ** 2, nan=0.0)
    if rho.size > 2:
        # the two points nearest the axis carry the noise of the coarse equilibrium grid
        grad_rho[:2], grad_rho_sq[:2] = grad_rho[2], grad_rho_sq[2]
    scal = run["scalars"]
    step = max(1, (eq["nw"] - 1) // (MAP_POINTS - 1))
    return {"rho": rho, "theta": 2.0 * np.pi * np.arange(k) / k,
            "R": r_pts, "Z": z_pts, "B": b_tot, "B_pol": b_pol,
            "q": prof["q"], "psi": prof["psi"], "volume": prof["volume"], "vpr": vpr,
            "area": prof["area"], "grad_rho": grad_rho, "grad_rho_sq": grad_rho_sq,
            "n_e": prof["n_e"],
            "R_grid": eq["R"][::step], "Z_grid": eq["Z"][::step],
            "psi_rz": eq["psirz"][::step, ::step].astype(np.float32),
            "psi_axis": float(eq["simag"]), "psi_lcfs": float(eq["sibry"]),
            "R_axis": float(eq["rmaxis"]), "Z_axis": float(eq["zmaxis"]),
            "psi_n_grid": np.linspace(0.0, 1.0, eq["nw"]), "F": eq["fpol"],
            "boundary": resample_closed(np.column_stack([eq["rbbbs"], eq["zbbbs"]]), RING_POINTS),
            "limiter": resample_closed(np.column_stack([eq["rlim"], eq["zlim"]]), RING_POINTS),
            "P_heat": scal["P_external_total"] / 1e6, "P_rad": abs(scal["P_radiation_e"]) / 1e6,
            "B_0": scal.get("B_0", spec["B0"]), "Ip": scal.get("Ip", spec["Ip"]),
            "Z_eff": spec["Z_eff"], "Z_imp": IMPURITY_CHARGE[spec["impurity"]],
            "A_i": scal.get("A_i", closure["A_i"]), "T_sep": spec["T_sep"]}


def publish(summary, name, workdir, export_dir):
    for pattern in FINAL_FILES:
        src = Path(workdir) / pattern.format(name=name)
        if src.is_file():
            shutil.copy2(src, Path(export_dir) / src.name)
