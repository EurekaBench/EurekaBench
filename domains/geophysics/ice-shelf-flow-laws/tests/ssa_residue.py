#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util  # EVAL DELTA
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np


RHO_ICE = 917.0      # kg m^-3
RHO_WATER = 1030.0   # kg m^-3
GRAVITY = 9.8        # m s^-2
G_REDUCED = GRAVITY * (1.0 - RHO_ICE / RHO_WATER)   # g' = g (1 - rho/rho_w)

SHELF_METHOD = {
    "Amery": "pinns",
    "LarsenC": "pinns",
    "RnFlch": "xpinns",
    "Ross": "xpinns",
}

AGENT_GOV_EQN = None
AGENT_FRONT_EQN = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ssa_residue.py",
        description="Reproduce the SSA equation-residue results of Wang et al., Science 387, 1219 (2025).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["iso", "aniso", "agent"],  # EVAL DELTA
                   help="isotropic (single mu) or anisotropic (mu_h, mu_v) inversion")
    # EVAL DELTA: agent-submission evaluation and the train/analysis split
    p.add_argument("--run-dir", default="",
                   help="agent run directory holding equations.py and interface.json")
    p.add_argument("--stage", default="all", choices=["all", "train", "analyze"],
                   help="train: stop after the weights are saved; analyze: reuse trained_params.pkl")
    p.add_argument("--baseline-summary", default="",
                   help="classical-iso residue_summary.json to compare the agent against")
    p.add_argument("--aniso-summary", default="",
                   help="paper-aniso residue_summary.json to compare the agent against")
    p.add_argument("--shelf", default="Amery", choices=sorted(SHELF_METHOD),
                   help="ice shelf to analyse")
    p.add_argument("--method", default="auto", choices=["auto", "pinns", "xpinns"],
                   help="'auto' picks pinns/xpinns from --shelf as in the paper")
    p.add_argument("--data-dir", default="data", help="directory holding the .mat datasets")
    p.add_argument("--out-dir", default="results", help="directory for all outputs")
    p.add_argument("--tag", default="", help="extra suffix for the run directory")

    # --- reproducibility ---
    p.add_argument("--seed", type=int, default=2134,
                   help="random seed for network init and point sampling")

    # --- network ---
    p.add_argument("--n-hl", type=int, default=6, help="number of hidden layers")
    p.add_argument("--n-unit", type=int, default=40, help="units per hidden layer")

    p.add_argument("--lw", required=True,
                   help="comma-separated loss weights (see above for the ordering)")

    p.add_argument("--n-pt", required=True,
                   help="comma-separated number of sampling points (see above)")

    # --- optimisation ---
    p.add_argument("--epoch1", type=int, required=True, help="Adam iterations")
    p.add_argument("--epoch2", type=int, required=True,
                   help="L-BFGS budget (the library uses epoch2/3 as max_iterations)")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")

    # --- residue analysis ---
    p.add_argument("--zone-frac", type=float, default=0.05,
                   help="transition-band fraction in the SM Sec. III.A zone criteria")
    p.add_argument("--shear-frac", type=float, default=0.2,
                   help="shear/transverse cut in the SM Sec. III.A compression criterion")

    # --- re-analysis without re-training ---
    p.add_argument("--params", default="",
                   help="path to a saved .pkl of trained weights; if given, training is skipped")

    p.add_argument("--float64", action="store_true",
                   help="run in float64 (slower; the paper used the JAX default float32)")
    return p


def flowline_strain_rates(u, v, u_x, u_y, v_x, v_y):
    theta = np.arctan2(v, u)
    c, s = np.cos(theta), np.sin(theta)

    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)          # tensor (not engineering) shear

    eps_ll = eps_xx * c * c + eps_yy * s * s + 2.0 * eps_xy * s * c
    eps_tt = eps_xx * s * s + eps_yy * c * c - 2.0 * eps_xy * s * c
    eps_lt = s * c * (eps_yy - eps_xx) + eps_xy * (c * c - s * s)
    return eps_ll, eps_lt, eps_tt


def zone_masks(eps_ll, eps_lt, eps_tt, zone_frac=0.05, shear_frac=0.2):
    valid = np.isfinite(eps_ll) & np.isfinite(eps_lt) & np.isfinite(eps_tt)

    # M = max |eps_ll| over the compressive part of the shelf
    compressive = valid & (eps_ll < 0)
    if compressive.any():
        m_scale = float(np.max(np.abs(eps_ll[compressive])))
    else:
        m_scale = float(np.max(np.abs(eps_ll[valid]))) if valid.any() else float("nan")
        print("[warn] no compressive (eps_ll < 0) points found -- the network is "
              "almost certainly under-trained. Falling back to M = max|eps_ll| over "
              "the whole domain; the compression zone will be empty and the zone "
              "statistics are NOT the paper's definition.", file=sys.stderr)

    thr = zone_frac * m_scale
    transition = valid & (np.abs(eps_ll) < thr)
    compression = (valid & (eps_ll < -thr)
                   & (np.abs(eps_ll) > shear_frac * np.abs(eps_lt))
                   & (np.abs(eps_ll) > shear_frac * np.abs(eps_tt)))
    extension = valid & (eps_ll > thr)
    return dict(valid=valid, compression=compression, transition=transition,
                extension=extension, m_scale=m_scale)


def relative_equation_error(e1, e2, e11, e12, e13, e21, e22, e23):
    import warnings
    with warnings.catch_warnings():
        # points outside the shelf are NaN in all three terms; that is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        den1 = np.nanmax(np.abs(np.stack([e11, e12, e13])), axis=0)
        den2 = np.nanmax(np.abs(np.stack([e21, e22, e23])), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel1 = np.abs(e1) / den1
        rel2 = np.abs(e2) / den2
    rel1[~np.isfinite(rel1)] = np.nan
    rel2[~np.isfinite(rel2)] = np.nan
    return rel1, rel2, den1, den2


def stats(a, mask):
    x = a[mask & np.isfinite(a)]
    if x.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "median", "p90", "p95", "p99", "max", "rms")} | {"n": 0}
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p90=float(np.percentile(x, 90)),
        p95=float(np.percentile(x, 95)),
        p99=float(np.percentile(x, 99)),
        max=float(np.max(x)),
        rms=float(np.sqrt(np.mean(x ** 2))),
    )


def domain_stress_scale(x, y, h_data):
    lx0 = (np.nanmax(x) - np.nanmin(x)) / 2.0
    ly0 = (np.nanmax(y) - np.nanmin(y)) / 2.0
    l0m = max(lx0, ly0)
    h0 = float(np.nanmean(h_data))
    return float(RHO_ICE * G_REDUCED * h0 ** 2 / l0m), h0, l0m


def analyse_residue(results, args):
    g = lambda k: np.asarray(results[k], dtype=np.float64)

    e1, e2 = g("e1"), g("e2")                       # residues f1, f2   [Pa]
    e11, e12, e13 = g("e11"), g("e12"), g("e13")    # I1, I2, I3        [Pa]
    e21, e22, e23 = g("e21"), g("e22"), g("e23")    # J1, J2, J3        [Pa]

    rel1, rel2, den1, den2 = relative_equation_error(e1, e2, e11, e12, e13, e21, e22, e23)

    # flow-line strain rates and the SM Sec. III.A zones
    eps_ll, eps_lt, eps_tt = flowline_strain_rates(
        g("u"), g("v"), g("u_x"), g("u_y"), g("v_x"), g("v_y"))
    masks = zone_masks(eps_ll, eps_lt, eps_tt, args.zone_frac, args.shear_frac)

    # hydrostatic stress scale sigma0
    sigma0, h0, l0m = domain_stress_scale(g("x"), g("y"), g("h_g"))
    if "scale" in results:                    # PINN: cross-check against the library's own value
        try:
            term0 = float(np.asarray(results["scale"]["term0"]).ravel()[0])
            if abs(term0 - sigma0) / term0 > 0.01:
                print(f"[warn] sigma0 recomputed ({sigma0:.4g} Pa) differs from "
                      f"DIFFICE_jax term0 ({term0:.4g} Pa) by "
                      f"{100*abs(term0-sigma0)/term0:.2f}%", file=sys.stderr)
            sigma0 = term0                     # prefer the library's exact value
        except Exception:
            pass

    fields = {
        "rel_err_1": rel1, "rel_err_2": rel2,
        "denom_1": den1, "denom_2": den2,
        "f1_norm": e1 / sigma0, "f2_norm": e2 / sigma0,
        "eps_ll": eps_ll, "eps_lt": eps_lt, "eps_tt": eps_tt,
        "mask_compression": masks["compression"].astype(np.int8),
        "mask_transition": masks["transition"].astype(np.int8),
        "mask_extension": masks["extension"].astype(np.int8),
    }

    zones = {"whole_shelf": masks["valid"],
             "compression": masks["compression"],
             "transition": masks["transition"],
             "extension": masks["extension"]}

    summary = {
        "sigma0_Pa": sigma0,
        "h0_m": h0,
        "L0_m": l0m,
        "eps_ll_compressive_max_per_s": masks["m_scale"],
        "n_valid_points": int(masks["valid"].sum()),
        "zones": {},
    }
    for zname, zmask in zones.items():
        summary["zones"][zname] = {
            "n_points": int(zmask.sum()),
            "area_fraction": float(zmask.sum() / max(masks["valid"].sum(), 1)),
            "rel_err_f1": stats(rel1, zmask),      # <-- the paper's "Err"
            "rel_err_f2": stats(rel2, zmask),
            "abs_f1_Pa": stats(np.abs(e1), zmask),
            "abs_f2_Pa": stats(np.abs(e2), zmask),
            "f1_over_sigma0": stats(np.abs(e1) / sigma0, zmask),
            "f2_over_sigma0": stats(np.abs(e2) / sigma0, zmask),
        }

    # viscosity summary (mu, or mu_h / mu_v for the anisotropic run)
    mu = g("mu")
    summary["viscosity"] = {"mu_Pa_s": stats(mu, masks["valid"])}
    if "eta" in results:
        eta = g("eta")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mu / eta
        summary["viscosity"]["mu_h_Pa_s"] = stats(mu, masks["valid"])
        summary["viscosity"]["mu_v_Pa_s"] = stats(eta, masks["valid"])
        summary["viscosity"]["mu_h_over_mu_v"] = stats(ratio, masks["valid"])
        summary["viscosity"]["mu_h_over_mu_v_extension"] = stats(ratio, masks["extension"])
        fields["mu_h_over_mu_v"] = ratio
    return summary, fields


def format_report(summary, args, elapsed):
    L = []
    A = L.append
    label = ("AGENT SUBMISSION" if args.mode == "agent" else  # EVAL DELTA
             "ISOTROPIC (single mu)" if args.mode == "iso" else "ANISOTROPIC (mu_h, mu_v)")
    resid = "f1, f2" if args.mode == "iso" else "f1^, f2^"
    A("=" * 78)
    A(f" SSA EQUATION RESIDUE -- {label}")
    A(f" shelf = {args.shelf}   method = {args.method}   seed = {args.seed}")
    A(f" residue symbol in the paper: {resid}")
    A("=" * 78)
    A("")
    A(f" hydrostatic stress scale  sigma0 = rho g' h0^2 / L0 = {summary['sigma0_Pa']:.4f} Pa")
    A(f"   with h0 = {summary['h0_m']:.2f} m,  L0 = {summary['L0_m']:.1f} m")
    A(f" valid grid points: {summary['n_valid_points']}")
    A(f" max |eps_ll| over compressive area: {summary['eps_ll_compressive_max_per_s']:.4e} 1/s")
    A("")
    A(" RELATIVE EQUATION ERROR   e = |f| / max(|I1|,|I2|,|I3|)      (SM Sec. IV.A, Fig. S14f)")
    A(" " + "-" * 76)
    A(f" {'zone':<13}{'area%':>7}{'mean':>10}{'median':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'max':>10}")
    for zname in ("whole_shelf", "compression", "transition", "extension"):
        z = summary["zones"][zname]
        s = z["rel_err_f1"]
        A(f" {zname:<13}{100*z['area_fraction']:>6.1f}%"
          f"{100*s['mean']:>9.2f}%{100*s['median']:>9.2f}%{100*s['p90']:>9.2f}%"
          f"{100*s['p95']:>9.2f}%{100*s['p99']:>9.2f}%{100*s['max']:>9.2f}%")
    A("   (rows above are for f1 / f1^;  f2 values are in residue_summary.json)")
    A("")
    A(" ABSOLUTE RESIDUE  |f1| [Pa]   and normalised |f1|/sigma0   (Fig. 4B colour scale)")
    A(" " + "-" * 76)
    A(f" {'zone':<13}{'mean[Pa]':>12}{'rms[Pa]':>12}{'max[Pa]':>12}{'mean/sig0':>12}{'rms/sig0':>12}")
    for zname in ("whole_shelf", "compression", "transition", "extension"):
        z = summary["zones"][zname]
        a, n = z["abs_f1_Pa"], z["f1_over_sigma0"]
        A(f" {zname:<13}{a['mean']:>12.4e}{a['rms']:>12.4e}{a['max']:>12.4e}"
          f"{n['mean']:>12.4e}{n['rms']:>12.4e}")
    A("")
    v = summary["viscosity"]
    if "mu_h_Pa_s" in v:
        A(" INFERRED ANISOTROPIC VISCOSITY")
        A(f"   mu_h : median {v['mu_h_Pa_s']['median']:.4e} Pa s")
        A(f"   mu_v : median {v['mu_v_Pa_s']['median']:.4e} Pa s")
        A(f"   mu_h/mu_v : median {v['mu_h_over_mu_v']['median']:.4f} (whole shelf), "
          f"{v['mu_h_over_mu_v_extension']['median']:.4f} (extension zone)")
    else:
        A(" INFERRED ISOTROPIC VISCOSITY")
        A(f"   mu : median {v['mu_Pa_s']['median']:.4e} Pa s")
    A("")
    A(" PAPER REFERENCE VALUES")
    A("   SM Sec. IV.A (Amery, isotropic) : e > 15% in the extension zone, < 1% elsewhere")
    A("   Fig. 4B (Ronne-Filchner)        : f1 (iso) Err > 10%;  f1^ (aniso) Err ~ 2.0%")
    A("")
    A(f" wall-clock: {elapsed/3600:.3f} h")
    A("=" * 78)
    return "\n".join(L)


def compare_with_counterpart(summary, args, run_root):
    other = "aniso" if args.mode == "iso" else "iso"
    name = run_dir_name(args.shelf, other, args.method, args.seed, args.tag)
    candidates = [
        run_root.parent / name,
        run_root.parent.parent / other / name,
        run_root.parent.parent.parent / other / name,
    ]
    f = next((c / "residue_summary.json" for c in candidates
              if (c / "residue_summary.json").exists()), None)
    if f is None:
        return ""
    try:
        with open(f) as fh:
            o = json.load(fh)
    except Exception:
        return ""
    L = ["", "=" * 78, " ISOTROPIC  vs  ANISOTROPIC   (relative equation error of f1)", "=" * 78]
    iso_s = summary if args.mode == "iso" else o["summary"]
    ani_s = o["summary"] if args.mode == "iso" else summary
    L.append(f" {'zone':<13}{'iso mean':>12}{'aniso mean':>12}{'reduction':>12}"
             f"{'iso p95':>12}{'aniso p95':>12}")
    for zname in ("whole_shelf", "compression", "transition", "extension"):
        i = iso_s["zones"][zname]["rel_err_f1"]
        a = ani_s["zones"][zname]["rel_err_f1"]
        red = i["mean"] / a["mean"] if a["mean"] > 0 else float("nan")
        L.append(f" {zname:<13}{100*i['mean']:>11.2f}%{100*a['mean']:>11.2f}%{red:>11.2f}x"
                 f"{100*i['p95']:>11.2f}%{100*a['p95']:>11.2f}%")
    L.append("")
    L.append(" Paper: the anisotropic residue in the extension zone is ~10x smaller (Fig. 4B).")
    L.append("=" * 78)
    return "\n".join(L)


def run_dir_name(shelf, mode, method, seed, tag):
    name = f"{shelf}_{method}_{mode}_seed{seed}"
    return name + (f"_{tag}" if tag else "")


def patch_debug_callback():
    try:
        from jax._src import debugging

        def debug_callback_impl_numpy(*args, callback, effect=None, **kwargs):
            callback(*(np.asarray(a) for a in args))
            return ()

        debugging.debug_callback_p.def_impl(debug_callback_impl_numpy)
        return True
    except Exception as exc:                                    # pragma: no cover
        print(f"[warn] could not apply the jax.debug.callback shim ({exc}). "
              "If the run stalls at 0% CPU during L-BFGS, this is why.",
              file=sys.stderr)
        return False


def patch_xpinn_eta_stitching():
    try:
        from diffice_jax.model.xpinns import prediction as xp

        orig = xp.redimensionalize

        def fixed(output, data_norm, data_info, idxgall, aniso=False):
            varsub, varsub_h, idxvars, idxvars_h = orig(output, data_norm, data_info, idxgall, aniso)
            return varsub, varsub_h, list(range(len(varsub))), idxvars_h

        xp.redimensionalize = fixed
        return True
    except Exception as exc:                                    # pragma: no cover
        print(f"[warn] could not patch the xpinn eta stitching bug ({exc}); "
              "anisotropic XPINN runs will report mu_v as a copy of mu_h.",
              file=sys.stderr)
        return False


def init_pinn_agent(parent_key, n_hl, n_unit, n_mu):
    from jax import random
    from diffice_jax.model.pinns.initialization import init_single_net
    layers1 = [2] + n_hl * [n_unit] + [3]
    layers2 = [2] + n_hl * [n_unit] + [n_mu]
    keys = random.split(parent_key, 2)
    params_u = init_single_net(keys[0], layers1)
    params_mu = init_single_net(keys[0], layers2)
    return [params_u, params_mu]


def init_xpinn_agent(parent_key, n_hl, n_unit, n_sub, n_mu):
    from jax import random
    from jax.tree_util import tree_map
    from diffice_jax.model.xpinns.initialization import init_single_net
    layers1 = [2] + n_hl * [n_unit] + [3]
    layers2 = [2] + n_hl * [n_unit] + [n_mu]
    _, *keys = random.split(parent_key, 2 * n_sub + 1)
    params_u = tree_map(lambda x: init_single_net(x, layers1), keys[0:n_sub])
    params_mu = tree_map(lambda x: init_single_net(x, layers2), keys[n_sub:])
    return dict(net_u=params_u, net_mu=params_mu)


# EVAL DELTA: the library returns only the first two viscosity columns
def visc_pair(out, first, n_unknown):
    cols = out[:, 3 + first:3 + min(first + 2, n_unknown)]
    return cols if cols.shape[1] == 2 else cols[:, [0, 0]]


def extra_viscosities(n_unknown, predict_call, shift_net):
    extra = []
    for first in range(2, n_unknown, 2):
        results = predict_call(shift_net(first))
        extra.append(np.asarray(results["mu"]))
        if first + 1 < n_unknown:
            extra.append(np.asarray(results["eta"]))
    return extra


def save_params(params, run_root):
    pkl_path = Path(run_root) / "trained_params.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(params, fh, pickle.HIGHEST_PROTOCOL)
    print(f"[info] trained weights saved to {pkl_path}", flush=True)
    return pkl_path


def load_ckpt(path):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return None


def dump_ckpt(path, obj):
    # write-then-rename so a job killed mid-write cannot corrupt the checkpoint
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def adam_opt_ckpt(key, lossf, params, dataf, epoch, lr, run_root,
                  aniso=False, schdul=None, ckpt_every=5000):
    import optax
    from jax import random
    from diffice_jax.optimizer.optimization import adam_minimizer

    if schdul is None:
        schdul = lambda x: 1.0
    wsp0 = lossf.wsp if hasattr(lossf, "wsp") else None

    opt_Adam = optax.adam(learning_rate=lr)
    opt_state = opt_Adam.init(params)
    loss_all = []
    start = 0

    ckpt_path = Path(run_root) / "adam_ckpt.pkl"
    ck = load_ckpt(ckpt_path)
    if ck is not None:
        params, opt_state, key = ck["params"], ck["opt_state"], ck["key"]
        start, loss_all = ck["step"], ck["loss_all"]
        if wsp0 is not None and ck["wsp"] is not None:
            lossf.wsp = ck["wsp"]
        print(f"[info] Adam checkpoint found -- resuming from step {start}/{epoch}", flush=True)
    if start >= epoch:
        return params, loss_all

    nc = int(round(epoch / 5))
    for step in range(start, epoch):
        key = random.split(key, 1)[0]
        data = dataf(key)
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        if (step + 1) % 100 == 0:
            print(f"Step: {step+1} | Loss: {loss_info[0]:.4e} | Loss_d: {loss_info[1]:.4e} |"
                  f" Loss_e: {loss_info[2]:.4e} | Loss_b: {loss_info[3]:.4e}", file=sys.stderr)
            if aniso:
                lossf.wsp = wsp0 * schdul(step + 1)
        loss_all.append(np.asarray(loss_info[0:4]))
        if (step + 1) % ckpt_every == 0:
            dump_ckpt(ckpt_path, dict(step=step + 1, params=params, opt_state=opt_state,
                                      key=key, loss_all=loss_all,
                                      wsp=(lossf.wsp if wsp0 is not None else None)))

    # library tail: keep stepping until the last loss beats the recent minimum
    lossend = np.array(loss_all[-nc:])[:, 0]
    lmin = lossend.min()
    llast = lossend[-1]
    while llast > lmin:
        key = random.split(key, 1)[0]
        data = dataf(key)
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        llast = float(loss_info[0])
        loss_all.append(np.asarray(loss_info[0:4]))
    dump_ckpt(ckpt_path, dict(step=epoch, params=params, opt_state=opt_state, key=key,
                              loss_all=loss_all, wsp=(lossf.wsp if wsp0 is not None else None)))
    return params, loss_all


def lbfgs_opt_ckpt(lossf, params, data, epoch, run_root, chunk_iters=2000):
    import jax.flatten_util as flat_utl
    from tensorflow_probability.substrates import jax as tfp
    from diffice_jax.optimizer.optimization import lbfgs_function

    max_total = int(epoch / 3)
    total, evals = 0, 0
    loss_all = []

    ckpt_path = Path(run_root) / "lbfgs_ckpt.pkl"
    ck = load_ckpt(ckpt_path)
    if ck is not None:
        params, total, evals, loss_all = ck["params"], ck["iters"], ck["evals"], ck["loss_all"]
        print(f"[info] L-BFGS checkpoint found -- resuming at iteration {total}/{max_total}", flush=True)

    while total < max_total:
        func = lbfgs_function(lossf, params, data)
        init_1d = flat_utl.ravel_pytree(params)[0]
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_1d,
            tolerance=1e-10, max_iterations=min(chunk_iters, max_total - total))
        params = func.update(results.position)
        n_it = int(results.num_iterations)
        total += n_it
        evals += int(results.num_objective_evaluations)
        loss_all += [np.asarray(l) for l in func.loss]
        dump_ckpt(ckpt_path, dict(params=params, iters=total, evals=evals, loss_all=loss_all))
        print(f"[info] L-BFGS: {total}/{max_total} iterations, {evals} evaluations "
              f"(converged={bool(results.converged)}, failed={bool(results.failed)})", flush=True)
        if bool(results.converged):
            break
        if n_it < 5:      # line search cannot make progress; a restart will not help
            break
    print(f" Total iterations: {evals}")
    return params, loss_all


def run_pinns(args, jnp, random, lax, rawdata, run_root):
    from diffice_jax import normdata_pinn, dsample_pinn
    from diffice_jax import vectgrad, ssa_iso, dbc_iso, ssa_aniso, dbc_aniso
    from diffice_jax import init_pinn, solu_pinn
    from diffice_jax import loss_iso_pinn, loss_aniso_pinn
    from diffice_jax import predict_pinn

    aniso = args.mode == "aniso"  # EVAL DELTA: agent submissions train with the iso loss
    if args.mode == "agent":  # EVAL DELTA: the submission's equations replace the library's
        gov_eqn, front_eqn = AGENT_GOV_EQN, AGENT_FRONT_EQN
    else:
        gov_eqn = ssa_aniso if aniso else ssa_iso
        front_eqn = dbc_aniso if aniso else dbc_iso

    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    keys = random.split(key, 4)

    n_pt = jnp.array(args.n_pt_list, dtype="int32")
    n_pt2 = n_pt * 2

    data_all = normdata_pinn(rawdata)
    scale = data_all[4][0:2]

    if args.mode == "agent":  # EVAL DELTA: head width = number of declared unknown fields
        params = init_pinn_agent(keys[0], args.n_hl, args.n_unit, args.agent_n_unknown)
    else:
        params = init_pinn(keys[0], args.n_hl, args.n_unit, aniso=aniso)
    pred_u = solu_pinn()

    dataf = dsample_pinn(data_all, n_pt)
    keys_adam = random.split(keys[1], 5)
    data = dataf(keys_adam[0])
    dataf_l = dsample_pinn(data_all, n_pt2)
    key_lbfgs = keys[2]

    eqn_all = (gov_eqn, front_eqn)
    loss_ctor = loss_aniso_pinn if aniso else loss_iso_pinn
    NN_loss = loss_ctor(pred_u, eqn_all, scale, args.lw_list)
    NN_loss.lref = NN_loss(params, data)[0]

    loss1, loss2 = [], []
    if args.params:
        with open(args.params, "rb") as fh:
            params = pickle.load(fh)
        print(f"[info] loaded trained weights from {args.params}; training skipped", flush=True)
    else:
        if aniso:
            wsp_schdul = lambda x: lax.max(10 ** (-x / args.epoch1 * 2), 0.0125)
            params, loss1 = adam_opt_ckpt(keys_adam[0], NN_loss, params, dataf, args.epoch1,
                                          args.lr, run_root, aniso=True, schdul=wsp_schdul)
        else:
            params, loss1 = adam_opt_ckpt(keys_adam[0], NN_loss, params, dataf, args.epoch1,
                                          args.lr, run_root)
        try:
            params, loss2 = lbfgs_opt_ckpt(NN_loss, params, dataf_l(key_lbfgs), args.epoch2, run_root)
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            print("[warn] L-BFGS with the authors' 2x sampling points ran out of GPU "
                  "memory; retrying with 1x points (same count as Adam). This deviates "
                  "from the authors' recipe.", file=sys.stderr, flush=True)
            params, loss2 = lbfgs_opt_ckpt(NN_loss, params, dataf(key_lbfgs), args.epoch2, run_root)
        save_params(params, run_root)

    f_u = lambda x: pred_u(params, x)
    f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]
    predict_aniso = aniso or (args.mode == "agent" and args.agent_n_unknown >= 2)  # EVAL DELTA
    results = predict_pinn((f_u, f_gu, gov_eqn), data_all, aniso=predict_aniso)
    if args.mode == "agent" and args.agent_n_unknown > 2:  # EVAL DELTA
        zero_eqn = lambda net, x, scale: (jnp.zeros((x.shape[0], 2)),
                                          jnp.zeros((x.shape[0], 7)))
        shift_net = lambda first: (
            lambda x: jnp.hstack([f_u(x)[:, 0:3],
                                  visc_pair(f_u(x), first, args.agent_n_unknown)]))
        results["agent_extra"] = extra_viscosities(
            args.agent_n_unknown,
            lambda net: predict_pinn((net, f_gu, zero_eqn), data_all, aniso=True),
            shift_net)
    return params, results, loss1, loss2


def run_xpinns(args, jnp, random, lax, rawdata, run_root):
    from jax.tree_util import tree_map
    from diffice_jax import normdata_xpinn, dsample_xpinn
    from diffice_jax import ssa_iso, dbc_iso, ssa_aniso, dbc_aniso
    from diffice_jax import init_xpinn, solu_xpinn
    from diffice_jax import loss_iso_xpinn, loss_aniso_xpinn
    from diffice_jax import predict_xpinn

    aniso = args.mode == "aniso"  # EVAL DELTA: agent submissions train with the iso loss
    if args.mode == "agent":  # EVAL DELTA: the submission's equations replace the library's
        gov_eqn, front_eqn = AGENT_GOV_EQN, AGENT_FRONT_EQN
    else:
        gov_eqn = ssa_aniso if aniso else ssa_iso
        front_eqn = dbc_aniso if aniso else dbc_iso

    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    keys = random.split(key, 4)

    n_pt = jnp.array(args.n_pt_list, dtype="int32")
    n_pt2 = jnp.array(n_pt * 2, dtype="int32")

    data_all, idxgall, posi_all, idxcrop_all = normdata_xpinn(rawdata)
    scale = tree_map(lambda x: data_all[x][4][0:2], idxgall)
    print(f"[info] XPINNs: {len(idxgall)} sub-regions", flush=True)

    if args.mode == "agent":  # EVAL DELTA: head width = number of declared unknown fields
        params = init_xpinn_agent(keys[0], args.n_hl, args.n_unit, len(idxgall), args.agent_n_unknown)
    else:
        params = init_xpinn(keys[0], args.n_hl, args.n_unit, n_sub=len(idxgall), aniso=aniso)
    solNN = solu_xpinn(scale)

    dataf = dsample_xpinn(data_all, idxgall, n_pt)
    keys_adam = random.split(keys[1], 5)
    data = dataf(keys_adam[0])
    dataf_l = dsample_xpinn(data_all, idxgall, n_pt2)
    key_lbfgs = keys[2]

    eqn_all = (gov_eqn, front_eqn)
    loss_ctor = loss_aniso_xpinn if aniso else loss_iso_xpinn
    NN_loss = loss_ctor(solNN, eqn_all, scale, idxgall, args.lw_list)
    NN_loss.lref = NN_loss(params, data)[0]

    loss1, loss2 = [], []
    if args.params:
        with open(args.params, "rb") as fh:
            params = pickle.load(fh)
        print(f"[info] loaded trained weights from {args.params}; training skipped", flush=True)
    else:
        if aniso:
            wsp_schdul = lambda x: lax.max(10 ** (-x / args.epoch1 * 2), 0.0125)
            params, loss1 = adam_opt_ckpt(keys_adam[0], NN_loss, params, dataf, args.epoch1,
                                          args.lr, run_root, aniso=True, schdul=wsp_schdul)
        else:
            params, loss1 = adam_opt_ckpt(keys_adam[0], NN_loss, params, dataf, args.epoch1,
                                          args.lr, run_root)
        try:
            params, loss2 = lbfgs_opt_ckpt(NN_loss, params, dataf_l(key_lbfgs), args.epoch2, run_root)
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            print("[warn] L-BFGS with the authors' 2x sampling points ran out of GPU "
                  "memory; retrying with 1x points (same count as Adam). This deviates "
                  "from the authors' recipe.", file=sys.stderr, flush=True)
            params, loss2 = lbfgs_opt_ckpt(NN_loss, params, dataf(key_lbfgs), args.epoch2, run_root)
        save_params(params, run_root)

    f_u = lambda x, idx: solNN[0](params, x, idx)
    predict_aniso = aniso or (args.mode == "agent" and args.agent_n_unknown >= 2)  # EVAL DELTA
    results = predict_xpinn((f_u, gov_eqn), data_all, posi_all, idxcrop_all, idxgall, aniso=predict_aniso)
    if args.mode == "agent" and args.agent_n_unknown > 2:  # EVAL DELTA
        zero_eqn = lambda net, x, scale: (jnp.zeros((x.shape[0], 2)),
                                          jnp.zeros((x.shape[0], 7)))
        shift_net = lambda first: (
            lambda x, idx: jnp.hstack(
                [f_u(x, idx)[:, 0:3],
                 visc_pair(f_u(x, idx), first, args.agent_n_unknown)]))
        results["agent_extra"] = extra_viscosities(
            args.agent_n_unknown,
            lambda net: predict_xpinn((net, zero_eqn), data_all, posi_all,
                                      idxcrop_all, idxgall, aniso=True),
            shift_net)
    return params, results, loss1, loss2


def main(argv=None):
    args = build_parser().parse_args(argv)

    # ---- resolve derived settings -------------------------------------
    if args.method == "auto":
        args.method = SHELF_METHOD[args.shelf]

    args.lw_list = [float(t) for t in args.lw.split(",") if t.strip() != ""]
    args.n_pt_list = [int(t) for t in args.n_pt.split(",") if t.strip() != ""]

    # EVAL DELTA: agent-mode gates -- any missing or non-conforming output scores 0
    args.agent_n_unknown = 0
    if args.mode == "agent":
        run_dir = Path(args.run_dir).expanduser().resolve()
        eval_out = Path(args.out_dir).expanduser().resolve()
        eval_out.mkdir(parents=True, exist_ok=True)
        failures = []
        for fname in ("equations.py", "interface.json"):
            f = run_dir / fname
            if not f.exists() or f.stat().st_size == 0:
                failures.append(f"missing or empty required output: {fname}")
        interface = None
        if not failures:
            try:
                with open(run_dir / "interface.json") as fh:
                    interface = json.load(fh)
                names = list(interface["field_names"])
                n1, n2 = int(interface["f1_terms"]), int(interface["f2_terms"])
                if names[:3] != ["u", "v", "h"]:
                    failures.append("interface.json field_names does not start with u, v, h")
                if len(names) < 4:
                    failures.append("interface.json declares no unknown fields")
                if (n1, n2) != (3, 3):
                    failures.append("the evaluation machinery expects 3 additive terms per "
                                    f"momentum equation; interface.json declares {n1}, {n2}")
            except Exception as exc:
                failures.append(f"interface.json does not parse: {exc}")
        if not failures:
            if args.float64:
                os.environ["JAX_ENABLE_X64"] = "True"
            try:
                spec = importlib.util.spec_from_file_location(
                    "agent_equations", run_dir / "equations.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for fn in ("gov_eqn", "front_eqn"):
                    if not callable(getattr(module, fn, None)):
                        failures.append(f"equations.py does not define {fn}")
            except Exception as exc:
                failures.append(f"equations.py failed to import: {type(exc).__name__}: {exc}")
        if failures:
            verdict_path = eval_out / f"eval_results_{args.shelf}.json"
            with open(verdict_path, "w") as fh:
                json.dump({"score": 0, "gate_failures": failures}, fh, indent=2)
            print(f"[eval] score 0: {failures}", flush=True)
            print(f"[eval] wrote {verdict_path}", flush=True)
            return 0
        global AGENT_GOV_EQN, AGENT_FRONT_EQN
        args.agent_n_unknown = len(interface["field_names"]) - 3
        AGENT_GOV_EQN = module.gov_eqn
        AGENT_FRONT_EQN = module.front_eqn
        args.agent_interface = interface

    mode_key = "iso" if args.mode == "agent" else args.mode  # EVAL DELTA: iso loss for agent mode
    n_lw_expected = {("pinns", "iso"): 2, ("pinns", "aniso"): 3,
                     ("xpinns", "iso"): 3, ("xpinns", "aniso"): 4}[(args.method, mode_key)]
    n_pt_expected = 4 if args.method == "pinns" else 5
    if len(args.lw_list) != n_lw_expected:
        raise SystemExit(f"--lw needs {n_lw_expected} values for {args.method}/{args.mode}, "
                         f"got {len(args.lw_list)}")
    if len(args.n_pt_list) != n_pt_expected:
        raise SystemExit(f"--n-pt needs {n_pt_expected} values for {args.method}, "
                         f"got {len(args.n_pt_list)}")

    prefix = "data_pinns_" if args.method == "pinns" else "data_xpinns_"
    data_path = Path(args.data_dir).expanduser().resolve() / f"{prefix}{args.shelf}.mat"
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path}")

    run_root = (Path(args.out_dir).expanduser().resolve()
                / run_dir_name(args.shelf, args.mode, args.method, args.seed, args.tag))
    run_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "analyze" and not args.params:  # EVAL DELTA
        pkl = run_root / "trained_params.pkl"
        if not pkl.exists():
            raise SystemExit(f"stage=analyze but no trained weights at {pkl}")
        args.params = str(pkl)

    # ---- JAX setup (must precede any jax import that touches config) ---
    if args.float64:
        os.environ["JAX_ENABLE_X64"] = "True"
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    from scipy.io import loadmat, savemat

    # MUST run before any training: see patch_debug_callback.__doc__.
    shim_ok = patch_debug_callback()
    eta_ok = patch_xpinn_eta_stitching()

    devices = jax.devices()
    print("=" * 78, flush=True)
    print(f" ssa_residue.py   mode={args.mode}  shelf={args.shelf}  method={args.method}", flush=True)
    print(f" jax {jax.__version__} | devices: {devices}", flush=True)
    print(f" output -> {run_root}", flush=True)
    print("=" * 78, flush=True)
    if not any(d.platform == "gpu" for d in devices):
        print("[warn] no GPU visible to JAX -- this will be extremely slow on CPU.", file=sys.stderr, flush=True)

    # ---- record the exact configuration -------------------------------
    import diffice_jax
    config = {
        "argv": sys.argv,
        "args": {k: v for k, v in vars(args).items()},
        "data_path": str(data_path),
        "data_bytes": data_path.stat().st_size,
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in devices],
        "diffice_jax": getattr(diffice_jax, "__version__", "1.0.2"),
        "diffice_jax_path": os.path.dirname(diffice_jax.__file__),
        "debug_callback_shim_applied": bool(shim_ok),
        "xpinn_eta_stitching_patch_applied": bool(eta_ok),
        "constants": {"rho_ice": RHO_ICE, "rho_water": RHO_WATER,
                      "g": GRAVITY, "g_reduced": G_REDUCED},
    }
    with open(run_root / "config.json", "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    # ---- load, train, predict -----------------------------------------
    print(f"[info] loading {data_path}", flush=True)
    rawdata = loadmat(str(data_path))

    t0 = time.time()
    runner = run_pinns if args.method == "pinns" else run_xpinns
    params, results, loss1, loss2 = runner(args, jnp, random, lax, rawdata, run_root)
    elapsed = time.time() - t0
    print(f"[info] training + prediction finished in {elapsed/3600:.3f} h", flush=True)

    if args.stage == "train":  # EVAL DELTA: analysis runs separately via --stage analyze
        print(f"[info] stage=train done; weights at {run_root / 'trained_params.pkl'}", flush=True)
        return 0

    pkl_path = run_root / "trained_params.pkl"

    # ---- residue analysis ---------------------------------------------
    agent_extra = results.pop("agent_extra", [])  # EVAL DELTA
    results_np = {}
    for k, v in results.items():
        if k == "scale":
            results_np[k] = {kk: np.asarray(vv) for kk, vv in v.items()}
        else:
            results_np[k] = np.asarray(v)

    try:
        summary, fields = analyse_residue(results_np, args)
    except Exception:
        import traceback
        traceback.print_exc()
        print("[error] residue analysis failed -- raw prediction fields are still "
              "being saved to fields.mat so nothing is lost.", file=sys.stderr)
        summary, fields = None, {}

    # ---- save everything ----------------------------------------------
    loss_all = (list(loss1) + list(loss2))
    out_mat = dict(results_np)
    for k, v in fields.items():
        out_mat[k] = v.astype(np.float32) if v.dtype.kind == "f" else v
    if args.mode == "agent":  # EVAL DELTA: store each field under its declared name
        declared = list(args.agent_interface["field_names"])[3:]
        arrays = [results_np.get("mu"), results_np.get("eta")] + list(agent_extra)
        for name, arr in zip(declared, arrays):
            if arr is not None:
                out_mat[name] = np.asarray(arr, dtype=np.float32)
    if loss_all:
        out_mat["loss"] = np.asarray(loss_all)
    savemat(str(run_root / "fields.mat"), out_mat, do_compression=True)
    if summary is None:
        print(f"[error] no residue summary produced; fields saved to "
              f"{run_root/'fields.mat'}", file=sys.stderr)
        return 1

    with open(run_root / "residue_summary.json", "w") as fh:
        json.dump({"config": config["args"], "summary": summary}, fh, indent=2)

    report = format_report(summary, args, elapsed)
    report += compare_with_counterpart(summary, args, run_root)
    with open(run_root / "residue_summary.txt", "w") as fh:
        fh.write(report + "\n")
    print(report, flush=True)
    written = [run_root / "fields.mat", run_root / "residue_summary.json",
               run_root / "residue_summary.txt"]
    if pkl_path.exists():          # absent when --params re-used an existing network
        written.append(pkl_path)
    print("\n[info] wrote:\n  " + "\n  ".join(str(p) for p in written), flush=True)

    if args.mode == "agent":  # EVAL DELTA: machine-readable verdict for the benchmark
        comparison = {}
        refs = {"classical_iso": args.baseline_summary, "paper_aniso": args.aniso_summary}
        for ref_name, ref_path in refs.items():
            if ref_path and Path(ref_path).exists():
                with open(ref_path) as fh:
                    base = json.load(fh)["summary"]
                comparison[ref_name] = {}
                for zname in ("whole_shelf", "compression", "transition", "extension"):
                    i = base["zones"][zname]["rel_err_f1"]
                    a = summary["zones"][zname]["rel_err_f1"]
                    comparison[ref_name][zname] = {
                        "ref_mean": i["mean"], "agent_mean": a["mean"],
                        "ref_p95": i["p95"], "agent_p95": a["p95"]}
        eval_out = Path(args.out_dir).expanduser().resolve()
        verdict_path = eval_out / f"eval_results_{args.shelf}.json"
        with open(verdict_path, "w") as fh:
            json.dump({"score": 1, "gate_failures": [],
                       "shelf": args.shelf,
                       "interface": args.agent_interface,
                       "residue_summary": str(run_root / "residue_summary.json"),
                       "comparison": comparison}, fh, indent=2)
        print(f"[info] wrote {verdict_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
