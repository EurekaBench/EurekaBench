import numpy as np
from pathlib import Path

EQUILIBRIA_DIR = Path(__file__).resolve().parent / "equilibria"
N_RHO = 25
MACHINE = dict(
    R_major=1.67, a_minor=0.6, B_0=2.0, Ip=0.9e6, elongation=1.7,
    impurity="C", Z_eff=2.0, n_line=4.0e19,
    T_bc=0.4, rho_bc=0.9, T_sep=0.1,
    P_aux=8.0e6, electron_heat_fraction=0.5, aux_location=0.0, aux_width=0.3,
)
SCAN_DELTAS = [round(float(d), 1) for d in np.linspace(-0.5, 0.5, 11)]
SETTLED_PROFILES = [
    "F", "FFprime", "Phi", "R_in", "R_out", "R_major_profile", "Ip_profile",
    "Ip_profile_from_geo", "area", "epsilon", "g0", "g0_over_vpr", "g1", "g1_over_vpr",
    "g1_over_vpr2", "g2", "g2g3_over_rhon", "g3", "gm4", "gm5", "gm9",
    "magnetic_shear", "n_e", "n_i", "n_impurity", "psi", "psi_from_Ip", "psi_from_geo",
    "psi_norm", "q", "r_mid", "spr", "volume", "vpr", "Z_eff", "Z_i", "Z_impurity",
]
SETTLED_SCALARS = [
    "A_i", "A_impurity", "B_0", "Ip", "R_major", "a_minor", "Phi_b", "rho_b", "drho",
    "drho_norm", "z_magnetic_axis", "q95", "q_min", "li3", "n_e_line_avg",
    "n_e_volume_avg", "n_i_line_avg", "n_i_volume_avg", "fgw_n_e_line_avg",
    "fgw_n_e_volume_avg", "P_aux_generic_i", "P_aux_generic_e", "P_aux_generic_total",
]
PROPS_SURFACE = ["R_surface", "Z_surface", "B_surface", "B_pol_surface"]
