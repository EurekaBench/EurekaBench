# CAMELS CV Simulations, Four Galaxy-Formation Models

The archive serves the CAMELS CV (cosmic variance) sets of four galaxy-formation models: `SIMBA`, `IllustrisTNG`, `Astrid`, and `Swift-EAGLE`. Each set contains 27 simulations named `CV_0` to `CV_26`. Every simulation evolves a periodic box of 25 Mpc/h on a side containing dark matter and gas from shortly after the Big Bang to today, at the fixed fiducial cosmological and astrophysical parameters of its model (recorded in the parameter table). The 27 initial conditions are shared across the four models: `CV_i` starts from the same initial density field in every model, so the same large-scale structure evolves under four different baryonic physics implementations. The models differ only in their hydrodynamics solvers and subgrid baryonic physics (gas cooling, star formation, stellar feedback, black-hole growth and feedback); the details of those implementations are not documented here — probing their consequences in the data is yours to do. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which of their data products to download; every quantitative analysis you must implement yourself from the downloaded files.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Every command takes `--suite` (one of `SIMBA`, `IllustrisTNG`, `Astrid`, `Swift-EAGLE`) and `--set_type CV`. Files are downloaded from the archive on first access and cached under `/workspace/camels_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table ~2KB, one power spectrum ~22KB, one catalogue 15-25MB, one snapshot 2-3GB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_suites` | (none) | the available suites and set |
| `list_simulations` | `--suite`, `--set_type` | names of all simulations |
| `get_params` | `--suite`, `--set_type` | path of the parameter table, its columns, row count |
| `get_power_spectrum` | `--suite`, `--set_type`, `--sim_name`, `--ptype` (default `m`), `--redshift` (default `0.0`) | path of the P(k) file, bin count, k-range |
| `get_group_catalog` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 catalogue, redshift, object counts, field names |
| `get_snapshot` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 snapshot, redshift, particle counts, field names |

## Data formats and conventions

- **Parameter table**: plain text, one row per simulation, first column the simulation name; the parameter values are identical across the 27 rows of a CV set (fiducial model), only the random seed column varies. `CV_i` uses the same initial-conditions seed in all four suites.
- **Power spectra**: two tab-separated columns, k in h/Mpc and P(k) in (Mpc/h)^3. `ptype` selects the density field: `m` total matter, `c` cold dark matter, `g` gas, `s` stars, `bh` black holes (the file name encodes the redshift; a missing combination returns an error).
- **Catalogues** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`. Masses are in 1e10 Msun/h (`*MassType` arrays are indexed by particle type: 0 gas, 1 dark matter, 4 stars, 5 black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields.
- **Snapshots** (HDF5): full particle data under `PartType0` (gas), `PartType1` (dark matter), `PartType4` (stars), `PartType5` (black holes), with a `Header` group. Units follow the catalogue conventions; exact particle fields differ slightly between models, so check the command output.
- **Snapshot numbering**: snapshots are numbered 000-090 from early times to today; `90` is z=0. Not every number is stored for every simulation; z=0 always is.

These six commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.