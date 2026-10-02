You are a geophysicist studying the flow of Antarctic ice shelves, the floating extensions of the ice sheet that buttress the grounded ice and slow its discharge into the ocean. The motion of an ice shelf is driven by gravity and modeled by the classical isotropic shallow-shelf approximation (SSA), with the depth-averaged momentum equations f1 = 0 and f2 = 0:

    f1 = ∂/∂x [ μh (4 ∂u/∂x + 2 ∂v/∂y) ] + ∂/∂y [ μh (∂u/∂y + ∂v/∂x) ] − ρ_i g (1 − ρ_i/ρ_w) h ∂h/∂x

    f2 = ∂/∂y [ μh (4 ∂v/∂y + 2 ∂u/∂x) ] + ∂/∂x [ μh (∂u/∂y + ∂v/∂x) ] − ρ_i g (1 − ρ_i/ρ_w) h ∂h/∂y

where u, v are the horizontal velocity components, h is the ice thickness, μ(x, y) is a single scalar effective viscosity field, ρ_i = 917 kg/m³ and ρ_w = 1030 kg/m³ are the densities of ice and seawater, and g = 9.8 m/s² is the gravitational acceleration.

The existing ice flow law was calibrated from laboratory experiments on polycrystalline synthetic ice. It has never been validated at the scale of real Antarctic ice shelves, so the rheology of real ice shelves may differ significantly from that of laboratory ice. Consequently, not all of the assumptions underlying the classical SSA equations necessarily hold for real ice shelves. It remains unclear which assumptions underlying the SSA equations break down for real ice shelves. 



Your task is to discover the mechanism that explains this phenomenon. Concretely, you need to identify which assumptions of the SSA equations are problematic for real ice shelves. Your goal is to derive new SSA equations that revise these assumptions so that they describe the flow of real Antarctic ice shelves better than the existing equations do. Derive your revised equations first, then use the provided ice-shelf forward simulator to check that they behave as you expect. Keep revising your equations in light of the tests until you are convinced that they resolve those problems.

Every claim in your new SSA equations needs to be backed by data you actually collected. Your new SSA equations are evaluated on satellite remote-sensing data of real Antarctic ice shelves. The goal is to minimize two errors when the observed ice velocity and thickness are used to infer the unknown fields in your equations: the residual of your equations, and the misfit between the fitted velocity and thickness and the observations. Your experimental findings will also be reviewed by an independent researcher who has access to the same simulator, to assess whether the mechanism you discover can yield new insights.

You need to produce three deliverables. If any of them is missing or does not follow the requirements below, your solution receives a score of 0.
Deliverable 1:
Write the executable form of your mechanism, your final SSA equations, to `/workspace/mechanisms/equations.py`, in exactly the format below. The file defines the two functions below:
  - `gov_eqn(net, x, scale)` -> `(f_eqn, val_term)`: your momentum equations over the domain.
  - `front_eqn(net, x, nb, scale)` -> `(f_eqn, val_term)`: your dynamic boundary conditions at the calving front (`nb`: outward unit normal).

`net` maps normalized coordinates of shape (N, 2) to the normalized fields, one column per field: u, v, h first, then the unknown fields of your equations. `f_eqn` of shape (N, 2) stacks the residues of your two equations in normalized form. `val_term` stacks the three additive terms of f1, then the three of f2, then the effective strain rate as the last column (seven columns in total). Use jax only; everything needs to be differentiable. The classical isotropic SSA in exactly this format:

```python
import jax.numpy as jnp
from jax import vjp, vmap, lax

def colgrad(func, x):
    out, pull = vjp(func, x)
    n = out.shape[1]
    seed = jnp.zeros([n, x.shape[0], n])
    for i in range(n):
        seed = seed.at[i, :, i].set(1.)
    g = vmap(pull, in_axes=0)(seed)[0]
    g = g.transpose(1, 0, 2).reshape(x.shape[0], x.shape[1] * n)
    return g, out

def gov_eqn(net, x, scale):
    dmean, drange = scale[0:2]
    lx0, ly0, u0, v0 = drange[0:4]
    umax = lax.max(u0, v0)
    lmax = lax.max(lx0, ly0)
    cu, cv = u0 / umax, v0 / umax
    cx, cy = lx0 / lmax, ly0 / lmax

    def fluxes(x):
        g, out = colgrad(net, x)
        h = out[:, 2:3]
        mu = out[:, 3:4]
        u_x = g[:, 0:1] * cu / cx
        u_y = g[:, 1:2] * cu / cy
        v_x = g[:, 2:3] * cv / cx
        v_y = g[:, 3:4] * cv / cy
        h_x = g[:, 4:5] / cx
        h_y = g[:, 5:6] / cy
        strate = (u_x**2 + v_y**2 + 0.25 * (u_y + v_x)**2 + u_x * v_y) ** 0.5
        q1 = 2 * mu * h * (2 * u_x + v_y)
        q2 = 2 * mu * h * (2 * v_y + u_x)
        q12 = mu * h * (u_y + v_x)
        p1 = h * h_x
        p2 = h * h_y
        return jnp.hstack([q1, q2, q12, p1, p2, strate])

    gq, q = colgrad(fluxes, x)
    f1t1 = gq[:, 0:1] / cx    # d(q1)/dx
    f1t2 = gq[:, 5:6] / cy    # d(q12)/dy
    f1t3 = q[:, 3:4]          # p1
    f2t1 = gq[:, 3:4] / cy    # d(q2)/dy
    f2t2 = gq[:, 4:5] / cx    # d(q12)/dx
    f2t3 = q[:, 4:5]          # p2
    strate = q[:, 5:6]
    f1 = f1t1 + f1t2 - f1t3
    f2 = f2t1 + f2t2 - f2t3
    f_eqn = jnp.hstack([f1, f2])
    val_term = jnp.hstack([f1t1, f1t2, f1t3, f2t1, f2t2, f2t3, strate])
    return f_eqn, val_term

def front_eqn(net, x, nb, scale):
    dmean, drange = scale[0:2]
    lx0, ly0, u0, v0 = drange[0:4]
    umax = lax.max(u0, v0)
    lmax = lax.max(lx0, ly0)
    cu, cv = u0 / umax, v0 / umax
    cx, cy = lx0 / lmax, ly0 / lmax
    g, out = colgrad(net, x)
    h = out[:, 2:3]
    mu = out[:, 3:4]
    u_x = g[:, 0:1] * cu / cx
    u_y = g[:, 1:2] * cu / cy
    v_x = g[:, 2:3] * cv / cx
    v_y = g[:, 3:4] * cv / cy
    b1 = 2 * mu * (2 * u_x + v_y)
    b2 = 2 * mu * (2 * v_y + u_x)
    b12 = mu * (u_y + v_x)
    bh = 0.5 * h
    f1 = b1 * nb[:, 0:1] + b12 * nb[:, 1:2] - bh * nb[:, 0:1]
    f2 = b12 * nb[:, 0:1] + b2 * nb[:, 1:2] - bh * nb[:, 1:2]
    return jnp.hstack([f1, f2]), jnp.hstack([b1, b2, b12, bh])
```

Deliverable 2:
Write `/workspace/mechanisms/interface.json`: `field_names` — the network output columns of your `equations.py` in order; `f1_terms`, `f2_terms` — the number of additive terms of each momentum equation in `val_term`. For the classical equations this would be:

```json
{
  "field_names": ["u", "v", "h", "mu"],
  "f1_terms": 3,
  "f2_terms": 3
}
```

This file needs to be consistent with your `equations.py`.

Deliverable 3:
Keep an experiment log at /workspace/mechanisms/experiment.log. Treat it as your laboratory notebook: it is the primary reference for the researchers who will continue this work, and you choose how to organize it. Record what they will need to follow and extend your investigation: the questions you pursued, the simulator configurations and analyses you ran, what you observed, the conclusions you drew, and the approaches that did not work and why. Give particular room to the findings that surprised you or that you found interesting along the way, whether or not they entered your final mechanism; these are often the most valuable part of a notebook for the next researcher. Move the useful evidence behind the log's findings into /workspace/mechanisms/: the figures, analysis outputs, small data files and simulation outputs your experiments produced in /workspace/tmp and /workspace/observations/.

You may also place other supporting evidence in /workspace/mechanisms/ and reference it from the files above, such as fitted-model diagnostics, ablation results, or small data tables that support your claims. Only this folder is kept when the session ends; everything else in the workspace, including /workspace/observations/, /workspace/tmp, and the simulation outputs you generated, is discarded. The independent researcher reviewing your mechanism sees exactly this folder. Everything under /workspace/mechanisms/ combined is limited to 1 GB; if the folder exceeds this limit, your solution receives a score of 0. Use this folder for final deliverables and their supporting evidence, not as scratch space; keep drafts and intermediate files in /workspace/tmp.

You have access to an H100 GPU and a time budget of 4 hours. You can use /workspace/tmp for notes and intermediate files.