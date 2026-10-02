# TORAX: a 1-D tokamak transport simulator, with a library of shapes

TORAX (Citrin et al., arXiv:2406.06718; github.com/google-deepmind/torax) integrates the
1-D flux-surface-averaged transport equations of a tokamak plasma in JAX. Nothing in it is
fitted to a machine: what a simulated plasma does follows from the geometry, the transport
model, the sources you switch on and the boundary conditions you impose. The geometry of a
run is one of the equilibria in the library, computed before any run from the boundary
shape and the current. Everything is reached through seven commands: `capabilities`,
`config_schema`, `shapes`, `props`, `geometry`, `simulate` and `read`.

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
configuration sections, the transport models / pedestal models / solvers / time-step
calculators / geometry types this build can select, the names of the shapes in the library,
and the resource limits.

## 2. Every field of a section: `config_schema`

`--section <name>`, one of the sections `capabilities` lists. Returns, for each variant of
that section, every field with its type and default. This is how you find out what you can
change; nothing is hidden and nothing is narrowed. Start here rather than guessing field
names.

## 3. The library of shapes: `shapes`

No arguments. Returns the machine (major radius, minor radius, field, current, elongation)
and, for every shape, its name, its boundary triangularity, the position of its magnetic
axis and the safety factor near the axis and at the boundary. The shapes run from a
triangularity of -0.5 to +0.5 in steps of 0.1 on the same machine and the same current,
each symmetric about the midplane; a shape and its mirror image differ in nothing but the
sign of the triangularity of the boundary. Shape names look like
`size1.00_kappa1.70_delta-0.50`.

## 4. The recorded runs

The recorded runs share one configuration and differ only in the shape they were given.
It is written out in full in section 7, and its ingredients are: deuterium with carbon as
the impurity at an effective charge of 2.0; the electron density prescribed and not evolved
as n_e(rho) = 1.25 n_line (1 - 0.6 rho^2) with n_line = 4.0e19 per cubic metre, given to
TORAX on 101 evenly spaced nodes; the ion and electron temperatures held at 0.4 keV at a
normalised radius of 0.9 by the `set_T_ped_n_ped` pedestal model and at 0.1 keV at the last
closed flux surface, with the profiles between the two prescribed by TORAX; 8 MW deposited
on axis with a Gaussian of width 0.3, half of it to the electrons, with ohmic heating and
the ion-electron exchange switched on and no radiation; the current profile held fixed;
the linear solver with predictor-corrector steps and the Pereverzev terms; the `chi`
time-step calculator with a prefactor of 200; and the `tglfnn-ukaea` transport model with
its conductivities bounded between 0.05 and 30 m^2/s. Inside a normalised radius of 0.1
the transport model is replaced by fixed coefficients (the `inner_patch` fields of the
transport section), because the surrogate is not trusted at the magnetic axis.

## 5. What a scored mechanism is handed: `props`

No arguments. Returns the entries a mechanism receives for one setting when it is scored,
grouped by their shape on the grid, together with the rule that decides what is in there
and the list of what is deliberately absent. Use it to write code that names the entries
correctly.

## 6. The geometry of a shape without running it: `geometry`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--shape` | required | a shape of the library |
| `--n_rho` | 25 | radial cells of the grid the surfaces are given on |
| `--n_theta` | 64 | poloidal angles per flux surface |
| `--output_name` | `geometry` | name of the JSON written |

Writes `rho` (the cell centres of a run with `n_rho` cells, with the axis and the last
closed flux surface added at its two ends), the poloidal flux `psi` in Wb at each of them,
the safety factor `q`, the poloidal angle grid `theta` about the magnetic axis, and the
arrays `R_surface`, `Z_surface` (m), `B_surface`, `B_pol_surface` (T) of shape
(`n_rho` + 2, `n_theta`): the position of every flux surface at every angle and the total
and poloidal field strength there. These are the surfaces a run on that shape reports,
before any plasma is put on them.

## 7. Running a simulation: `simulate`

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
leave out takes its TORAX defaults. Ask `config_schema` for the fields of any of them. A
geometry of type `eqdsk` names one of the library's shapes in `geometry_file`
(`size1.00_kappa1.70_delta-0.50.eqdsk`); the other geometry types of TORAX are open too,
but only the library carries the shapes the task is about.

Returns the path of the saved `.nc`, `sim_error`, the number of timesteps, the physics time
actually reached, the wall clock, and the variables available in each output group.

**Limits**: `numerics.t_final` ≤ 100 s, `geometry.n_rho` ≤ 200, `max_steps` ≤ 200000.

The configuration of the recorded runs, here on the reversed shape at -0.5, with the 101
density nodes abbreviated:

```json
{"plasma_composition": {"main_ion": {"D": 1.0}, "impurity": "C", "Z_eff": 2.0},
 "profile_conditions": {"Ip": 900000.0, "T_i": {"0.0": {"0.0": 2.0, "1.0": 0.1}},
   "T_i_right_bc": 0.1, "T_e": {"0.0": {"0.0": 2.0, "1.0": 0.1}}, "T_e_right_bc": 0.1,
   "n_e": {"0.0": {"0.0": 5.0e19, "...": "...", "1.0": 2.0e19}}, "n_e_right_bc": 2.0e19,
   "normalize_n_e_to_nbar": false},
 "numerics": {"t_final": 1.2, "evolve_ion_heat": true, "evolve_electron_heat": true,
   "evolve_density": false, "evolve_current": false, "max_dt": 0.1,
   "chi_timestep_prefactor": 200, "dt_reduction_factor": 3},
 "geometry": {"geometry_type": "eqdsk", "geometry_file": "size1.00_kappa1.70_delta-0.50.eqdsk",
   "cocos": 11, "Ip_from_parameters": false, "n_rho": 25, "n_surfaces": 100,
   "last_surface_factor": 0.99},
 "sources": {"ei_exchange": {}, "ohmic": {},
   "generic_heat": {"gaussian_location": 0.0, "gaussian_width": 0.3, "P_total": 8.0e6,
     "electron_heat_fraction": 0.5}},
 "pedestal": {"model_name": "set_T_ped_n_ped", "set_pedestal": true, "T_i_ped": 0.4,
   "T_e_ped": 0.4, "n_e_ped": 3.2e19, "rho_norm_ped_top": 0.9},
 "solver": {"solver_type": "linear", "use_predictor_corrector": true,
   "n_corrector_steps": 10, "use_pereverzev": true, "chi_pereverzev": 30,
   "D_pereverzev": 15},
 "time_step_calculator": {"calculator_type": "chi"},
 "transport": {"model_name": "tglfnn-ukaea", "chi_min": 0.05, "chi_max": 30.0,
   "D_e_min": 0.05, "apply_inner_patch": true, "rho_inner": 0.1, "chi_i_inner": 1.0,
   "chi_e_inner": 1.0, "D_e_inner": 0.2, "V_e_inner": 0.0}}
```

## 8. Reading a finished run: `read`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--file` | required | the `.nc` `simulate` returned, by name |
| `--variables` | all | JSON list of variable names |
| `--time_index` | -1 | which timestep, -1 for the last |
| `--output_name` | `read1` | name of the JSON written |

Writes the requested variables to `/workspace/observations/<output_name>.json`. The `.nc`
is an xarray DataTree and you can also open it directly with `xarray.open_datatree` if you
prefer to work in Python.

## 9. What a run contains

Three groups, on the coordinates `time`, `rho_norm` (cell grid plus boundaries),
`rho_cell_norm` and `rho_face_norm`:

- **`numerics`**: `sim_error`, `sim_status`, solver iteration counts, `sawtooth_crash`.
- **`profiles`**: the radial profiles. Kinetic (`T_e`, `T_i`, `n_e`, `n_i`, `n_impurity`,
  `pressure_thermal_*`), magnetic (`q`, `magnetic_shear`, `psi`, `j_total`,
  `j_bootstrap`), geometric (`volume`, `vpr`, `area`, `g0`–`g3`, `g1_over_vpr`,
  `epsilon`, `elongation`), the transport coefficients (`chi_turb_i`, `chi_turb_e`,
  `D_turb_e`, `V_turb_e`, the neoclassical `chi_neo_*`, `D_neo_e`, `V_neo_e`), and the
  source densities (`p_generic_heat_i/e`, `p_ohmic_e`, `ei_exchange`, and the radiation
  terms when they are switched on).
- **`scalars`**: the global quantities. `tau_E`, `W_thermal_total`, `P_heat_total`,
  `P_aux_total`, `P_ohmic_e`, `P_SOL_total`, `H98`, `H89P`, `dW_thermal_dt_smoothed`,
  `beta_N`, `q95`, `q_min`, `li3`, `n_e_line_avg`, `n_e_volume_avg`, `T_e_volume_avg`,
  `T_i_volume_avg`, `Ip`, `B_0`, `R_major`, `a_minor`, `A_i`.

`simulate` returns the exact variable list of the run it just did, so check there rather
than assuming a name exists: which variables appear depends on which sources and which
transport model the configuration switched on.

## 10. Transport models

`capabilities` lists what this build can select. They differ in what physics they contain:
a constant-coefficient placeholder, a critical-gradient model, a semi-empirical
Bohm/gyro-Bohm scaling, and neural-network surrogates of quasilinear gyrokinetic codes.
`config_schema --section transport` gives every field of every one of them, including the
bounds `chi_min`, `chi_max`, `D_e_min`, `D_e_max`, `V_e_min`, `V_e_max`, the inner and
outer transport patches, and the radial range `rho_min`, `rho_max` over which the model
acts. The recorded runs used `tglfnn-ukaea`, which reads the local gradients, the safety
factor and its shear, the collisionality and the shape of each flux surface.

Whether a coefficient sits on one of those bounds is worth checking: a bound that is
reached is an extrapolation rather than a prediction, and because the `chi` time-step
calculator sets `dt` from the largest `chi` on the grid, one large value can make a run
far slower than the rest.

## 11. Cost

A run is a few seconds of arithmetic behind roughly a minute of JAX tracing and
compilation, which is paid once per distinct configuration structure and then reused.
Changing a numerical value or the shape is cheap; changing which model or which equations
are solved triggers a fresh compilation. The recorded runs took 15 to 40 seconds each.
`simulate` returns the wall clock and the number of timesteps it took, which together tell
you whether a configuration is expensive because the model is expensive or because the
timestep collapsed.