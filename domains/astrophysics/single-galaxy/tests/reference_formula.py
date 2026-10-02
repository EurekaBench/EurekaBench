import numpy as np
from scipy.special import expit

PAPER = "arXiv:2602.07651"

G = 4.300917270e-6      # kpc Msun^-1 (km/s)^2
H0 = 0.067              # (km/s)/kpc, the value the paper quotes
OMEGA_B = 0.049
HUBBLE = 0.6711
Z_SUN = 0.0127
KAPPA = 2.0             # V_eff^2 = Vmax^2 + KAPPA * VelDisp^2
ALPHA_K = 0.45          # Eq. 6, latent variable / (1+z)^ALPHA_K
ALPHA_GAMMA = 0.33      # Eq. 6, compactness term * (1+z)^ALPHA_GAMMA

PROPERTIES = ["SubhaloMassType", "SubhaloVmax", "SubhaloVelDisp",
              "SubhaloStarMetallicity", "SubhaloHalfmassRadType",
              "SubhaloHalfmassRad"]

COEFFS = {
    "IllustrisTNG": (5.68, 0.62, 0.0024),
    "Astrid": (4.71, 0.61, 0.005),
    "SIMBA": (14.88, 0.62, -0.0014),
    "Swift-EAGLE": (8.58, 0.66, 0.0035),
}


def latent(props):
    mass_type = np.asarray(props["SubhaloMassType"], dtype=np.float64)
    m_star = np.maximum(mass_type[:, 4], 1e-12) * 1e10 / HUBBLE
    v_eff = np.sqrt(np.asarray(props["SubhaloVmax"], dtype=np.float64) ** 2
                    + KAPPA * np.asarray(props["SubhaloVelDisp"], dtype=np.float64) ** 2)
    z_star = np.maximum(np.asarray(props["SubhaloStarMetallicity"],
                                   dtype=np.float64), 0.0) / Z_SUN
    return z_star / m_star * (OMEGA_B * v_eff ** 3 / (G * H0)) * OMEGA_B


def compactness(props):
    r_star = np.maximum(np.asarray(props["SubhaloHalfmassRadType"],
                                   dtype=np.float64)[:, 4], 1e-8)
    r_total = np.maximum(np.asarray(props["SubhaloHalfmassRad"], dtype=np.float64), 1e-8)
    return r_star / r_total


def predict_omega_m(props, suite, redshift=0.0):
    k, c0, a0 = COEFFS[suite]
    x = latent(props) / (k * (1.0 + redshift) ** ALPHA_K)
    return np.log(expit(x) + c0) - a0 * (1.0 + redshift) ** ALPHA_GAMMA / compactness(props)
