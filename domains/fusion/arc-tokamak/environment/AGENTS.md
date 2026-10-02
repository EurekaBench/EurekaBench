# TORAX: a 1-D tokamak transport simulator

TORAX (Citrin et al., arXiv:2406.06718; github.com/google-deepmind/torax) integrates the
1-D flux-surface-averaged transport equations of a tokamak plasma in JAX. Nothing in it is
fitted to a machine: what a simulated plasma does follows from the geometry, the transport
model, the sources you switch on and the boundary conditions you impose. Everything is
reached through four commands: `capabilities` (what this build offers), `config_schema`
(every field of a configuration section), `simulate` (integrate) and `read` (pull numbers
out of a finished run).

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
model is expensive or because the timestep collapsed.