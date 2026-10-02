# TORAX, TokaMaker and GPEC: a coupled model of the tokamak edge

TORAX (Citrin et al., arXiv:2406.06718; github.com/google-deepmind/torax) integrates the
1-D flux-surface-averaged transport equations of a tokamak plasma in JAX. Nothing in it is
fitted to a machine: what a simulated plasma does follows from the geometry, the transport
model, the sources you switch on and the boundary conditions you impose. TokaMaker, from
the Open FUSION Toolkit, solves the free boundary Grad-Shafranov equilibrium of a shape you
specify and writes it as a gEQDSK. GPEC evaluates the infinite-n ideal ballooning stability
of every flux surface of a gEQDSK. The coupled model runs the three together for a machine
and an operating point and reports what its edge became. Everything is reached through
seven commands: `capabilities` (what this build offers), `config_schema` (every field of a
TORAX configuration section), `simulate` (integrate), `read` (pull numbers out of a
finished run), `equilibrium` (build an equilibrium), `ballooning` (its stability boundary)
and `run` (the coupled model).

## Usage

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Configurations are JSON. Pass them inline or, far easier for anything real, as a path to a
`.json` file you wrote:

```bash
python /workspace/bb_cli.py simulate --config /workspace/tmp/my_case.json --output_name case1
```

Files the commands write go to `/workspace/observations/`.

## 1. What this build offers: `capabilities`

No arguments. Returns the TORAX version, the JAX backend and devices, the list of
configuration sections, the transport models / pedestal models / solvers /
time-step calculators / geometry types this build can select, and the resource limits.
`external_models` reports transport models that need an external code, and says why one is
unavailable when it is.

## 2. Every field of a section: `config_schema`

`--section <name>`, one of the sections `capabilities` lists. Returns, for each variant of
that section, every field with its type and default. This is how you find out what you can
change; nothing is hidden and nothing is narrowed. Start here rather than guessing field
names.

## 3. Running a simulation: `simulate`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--config` | required | the whole TORAX configuration, JSON string or `.json` path |
| `--output_name` | `run1` | name of the output file (letters, digits, `.`, `_`, `-`) |
| `--progress_bar` | off | show TORAX's progress bar |
| `--log_timestep_info` | off | log every timestep, which shows a collapsing `dt` |
| `--max_steps` | 0 | 0 for no cap, otherwise stop after that many steps |
| `--timeout` | 3600 | seconds the CLI waits for the server |

The configuration is TORAX's own, unmodified. Its sections are `profile_conditions`,
`numerics`, `plasma_composition`, `geometry`, `sources`, `neoclassical`, `solver`,
`transport`, `pedestal`, `mhd`, `edge`, `time_step_calculator` and `restart`. A section you
leave out takes its TORAX defaults. Ask `config_schema` for the fields of any of them.

Returns the path of the saved `.nc`, `sim_error`, the number of timesteps, the physics time
actually reached, the wall clock, and the variables available in each output group.

The `geometry` section also accepts a gEQDSK written by `equilibrium`, given by name as its
`geometry_file` with `geometry_type` `eqdsk`; `config_schema --section geometry` lists the
fields that go with it.

**Limits**: `numerics.t_final` ≤ 100 s, `geometry.n_rho` ≤ 200, `max_steps` ≤ 200000.

A minimal configuration, for orientation:

```json
{"plasma_composition": {"main_ion": {"D": 0.5, "T": 0.5}, "impurity": "Ne", "Z_eff": 1.5},
 "profile_conditions": {"Ip": 12.0e6, "T_i": {"0.0": {"0.0": 20.0, "1.0": 1.0}},
   "T_i_right_bc": 1.0, "T_e": {"0.0": {"0.0": 20.0, "1.0": 1.0}}, "T_e_right_bc": 1.0,
   "n_e": {"0.0": {"0.0": 3.0e20, "1.0": 2.1e20}}, "n_e_right_bc": 2.1e20},
 "numerics": {"t_final": 12.0, "evolve_density": true},
 "geometry": {"geometry_type": "circular", "R_major": 4.62, "a_minor": 1.18,
   "B_0": 11.4, "elongation_LCFS": 1.8, "n_rho": 25},
 "sources": {"fusion": {}, "ei_exchange": {}, "ohmic": {},
   "generic_heat": {"P_total": 20.0e6, "gaussian_width": 0.1}},
 "pedestal": {"model_name": "set_P_ped_n_ped", "set_pedestal": true, "P_ped": 4.5e5,
   "n_e_ped": 2.1e20, "T_i_T_e_ratio": 1.0, "rho_norm_ped_top": 0.93},
 "solver": {"solver_type": "linear"},
 "time_step_calculator": {"calculator_type": "chi"},
 "transport": {"model_name": "qlknn"}}
```

## 4. Reading a finished run: `read`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--file` | required | the `.nc` `simulate` returned, by name |
| `--variables` | all | JSON list of variable names |
| `--time_index` | -1 | which timestep, -1 for the last |
| `--output_name` | `read1` | name of the JSON written |

Writes the requested variables to `/workspace/observations/<output_name>.json`. The `.nc`
is an xarray DataTree and you can also open it directly with `xarray.open_datatree` if you
prefer to work in Python.

## 5. What a run contains

Three groups, on the coordinates `time`, `rho_norm` (cell grid plus boundaries),
`rho_cell_norm` and `rho_face_norm`:

- **`numerics`**: `sim_error`, `sim_status`, solver iteration counts, `sawtooth_crash`.
- **`profiles`**: the radial profiles. Kinetic (`T_e`, `T_i`, `n_e`, `n_i`, `n_impurity`,
  `pressure_thermal_*`), magnetic (`q`, `magnetic_shear`, `psi`, `j_total`,
  `j_bootstrap`), geometric (`volume`, `vpr`, `area`, `g0`–`g3`, `epsilon`, `elongation`),
  the transport coefficients (`chi_turb_i`, `chi_turb_e`, `D_turb_e`, `V_turb_e`, the
  neoclassical `chi_neo_*`, `D_neo_e`, `V_neo_e`, `V_neo_ware_e`, and per-mode
  `chi_itg_*`, `chi_tem_*`, `chi_etg_e` when the model reports them), and the source
  densities (`p_alpha_i`, `p_alpha_e`, `p_ohmic_e`, `p_generic_heat_i/e`,
  `p_bremsstrahlung_e`, `p_cyclotron_radiation_e`, `p_impurity_radiation_e`,
  `ei_exchange`, `s_gas_puff`, `s_pellet`).
- **`scalars`**: the global quantities. `P_fusion`, `Q_fusion`, `P_alpha_total`,
  `P_SOL_total`, `P_LH`, `P_radiation_e`, `H98`, `H89P`, `tau_E`, `W_thermal_total`,
  `dW_thermal_dt_smoothed`, `beta_N`, `q95`, `q_min`, `li3`, `n_e_line_avg`,
  `n_e_volume_avg`, `T_e_volume_avg`, `T_i_volume_avg`, `fgw_n_e_line_avg`,
  `f_bootstrap`, `Ip`, `B_0`, `R_major`, `a_minor`.

`simulate` returns the exact variable list of the run it just did, so check there rather
than assuming a name exists: which variables appear depends on which sources and which
transport model the configuration switched on.

## 6. Transport models

`capabilities` lists what this build can select. They differ in what physics they contain:
a constant-coefficient placeholder, a critical-gradient model, a semi-empirical
Bohm/gyro-Bohm scaling, neural-network surrogates of quasilinear gyrokinetic codes, and,
where the external code is installed, the quasilinear codes themselves. `config_schema
--section transport` gives every field of every one of them, including the bounds
`chi_min`, `chi_max`, `D_e_min`, `D_e_max`, `V_e_min`, `V_e_max`, the inner and outer
transport patches, and the radial range `rho_min`, `rho_max` over which the model acts.

Whether a coefficient sits on one of those bounds is worth checking: a bound that is
reached is an extrapolation rather than a prediction, and because the `chi` time-step
calculator sets `dt` from the largest `chi` on the grid, one clipped point can make a run
far slower than the rest.

## 7. Cost

A run is a few seconds of arithmetic behind roughly ten seconds of JAX tracing and
compilation, which is paid once per distinct configuration structure and then reused.
Changing a numerical value is cheap; changing which model or which equations are solved
triggers a fresh compilation. Models that call an external code are slower than the
in-process ones by orders of magnitude. `simulate` returns the wall clock and the number of
timesteps it took, which together tell you whether a configuration is expensive because the
model is expensive or because the timestep collapsed. An `equilibrium` solve takes of the
order of ten seconds. The first `ballooning` call pays Julia's compilation, which takes
minutes; later calls are much faster.

## 8. Building an equilibrium: `equilibrium`

TokaMaker solves a free boundary equilibrium for the cross-section you specify and writes
it as a gEQDSK to `/workspace/observations/<output_name>.geqdsk`, with a JSON summary of
the same name beside it.

| Argument | Default | Meaning |
|----------|---------|---------|
| `--config` | required | the shape and the operating point, JSON string or `.json` path |
| `--output_name` | `equilibrium1` | name of the gEQDSK and the summary |

Configuration keys: `R0`, `Z0`, `a` (m) and `kappa`, `delta` set the boundary, with
`kappa_lower` and `delta_lower` for the lower half when it differs from the upper; `Ip` the
plasma current in A, `B0` the vacuum field at `R0` in T, `p_axis` the pressure on axis in
Pa; `pp` and `ffp` the two exponents of the power law flux functions for p' and FF', default
`[1.5, 1.5]`; `x_point`, `lower` for a lower single null (the default) or `none` for a
limited plasma; `machine_scale`, which multiplies every coordinate of the machine, default
1; `kinetic` a dict of `n_e` (m^-3), `T_e` and `T_i` (eV), `Z_eff` and `Z_imp` on a uniform
normalised poloidal flux grid, which replaces the flux functions by a bootstrap consistent
solve; `n_boundary`, `n_psi`, `maxits`, `urf` (under-relaxation of the nonlinear iteration, 0.2), `nl_tol` (its tolerance, 1e-6), `coil_weight`, `coil_bound_factor`, `isoflux_weight`, `coil_targets`
(a dict of coil currents in A, by coil name, the fit is pulled towards instead of the reference discharge's) and `coil_vsc` for the numerics. The
mesh, the coils and the vessel are those of the machine this build carries; `capabilities`
names the mesh file.

Returns the two file paths, the solver error flag and the global quantities TokaMaker
reports (`Ip`, `beta_pol`, `beta_n`, `l_i`, `q_95`). The summary holds the q profile, the
flux functions, the pressure and the boundary points.

## 9. Ballooning stability of an equilibrium: `ballooning`

GPEC computes, on every flux surface of a gEQDSK, the normalised pressure gradient `alpha`
of the equilibrium and the first infinite-n ideal ballooning boundary `alpha_critical`, and
writes them with `psi` to `/workspace/observations/<output_name>.json`.

| Argument | Default | Meaning |
|----------|---------|---------|
| `--file` | required | the gEQDSK, by name |
| `--both` | off | also the second boundary, `alpha_critical2`; slower |
| `--max_alpha_scale` | 8 | the search for the boundaries stops at this multiple of the equilibrium `alpha` of each surface |
| `--output_name` | `ballooning1` | name of the JSON written |
| `--timeout` | 3600 | seconds the calculation may take |

A surface where no boundary is found within the scan is written as `null`.

## 10. The coupled model: `run`

One setting of the coupled model: the equilibrium of the shape, the edge that forms in it
and the plasma TORAX evolves inside it. It writes `/workspace/observations/<output_name>.json`
with the five edge quantities under `truth`, the profiles, the global quantities and the
same `props` entries the evaluation reads, beside the final equilibrium
`<output_name>_eq.geqdsk` and the TORAX run `<output_name>_run.nc`.

| Argument | Default | Meaning |
|----------|---------|---------|
| `--config` | required | the machine and the operating point, JSON string or `.json` path |
| `--output_name` | `run1` | name of the files written |
| `--timeout` | 7200 | seconds the setting may take |

Configuration keys: `R0`, `Z0`, `a`, `kappa`, `delta`, `kappa_lower`, `delta_lower` and
`machine_scale` as for `equilibrium`, with the X-point at the bottom; `Ip` in A, `B0` in T,
`n_e_ped` the pedestal density in m^-3 of the prescribed profile, `P_heat` in W, `Z_eff`,
`T_sep` in eV, and `heating_location`, `heating_width`, `electron_heat_fraction` for the
deposition. A setting takes several minutes: it solves equilibria, evaluates their
stability and runs TORAX twice.