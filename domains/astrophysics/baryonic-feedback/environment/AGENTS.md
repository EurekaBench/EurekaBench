# DREAMS CDM Milky-Way Zoom-in Simulations

The archive serves the DREAMS CDM Milky-Way zoom-in suite: 1,024 hydrodynamic simulations, named as `list_simulations` reports, and a gravity-only twin of every one. Each simulation re-simulates one Milky-Way-mass system at high resolution inside a larger cosmological volume with the IllustrisTNG galaxy-formation model, from shortly after the Big Bang to today. Each box has its own initial density field and its own values of the five varied parameters, recorded in the parameter table; the numerical setup is common to all. The twin of a box evolves the same initial conditions with gravity alone, all matter collisionless. The simulations are pre-computed: you cannot run new ones.

## Varied parameters

- `Om`: the total matter density Omega_m.
- `s8`: the amplitude of the initial density fluctuations sigma_8.
- `SN1`: the supernova wind energy amplitude.
- `SN2`: the supernova wind speed amplitude.
- `BHFF`: the black hole feedback efficiency amplitude.

`SN1`, `SN2` and `BHFF` are multiplicative factors on the fiducial IllustrisTNG model, where 1 is fiducial. The baryon density and the Hubble constant are the same in all simulations (Omega_b = 0.046, h = 0.6909). A twin shares the parameters of its box.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Files are downloaded from the archive on first access and cached under `/workspace/dreams_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table tens of KB, one catalog 2-10 MB, one twin snapshot ~0.3 GB, one hydrodynamic snapshot ~3.5 GB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_simulations` | (none) | the suite path and the names of all simulations |
| `get_params` | (none) | path of the parameter table, its columns, row count |
| `get_group_catalog` | `--sim_name`, `--snapshot` (default `90`), `--kind` (`hydro` or `nbody`, default `hydro`) | path(s) of the HDF5 Subfind catalog, redshift, object counts, field names |
| `get_snapshot` | `--sim_name`, `--snapshot` (default `90`), `--kind` | path of the HDF5 particle snapshot, redshift, particle counts per type, mass table, field names |

## Data formats and conventions

- **Parameter table**: plain text, a header line naming the columns, then one row per simulation in box order.
- **Catalogs** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`, in the standard Subfind schema. Masses are in 1e10 Msun/h (`*MassType` and `*LenType` arrays are indexed by particle type: 0 gas, 1 high-resolution dark matter, 2 low-resolution boundary dark matter from outside the zoom region, 4 stars, 5 black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields.
- **Snapshots** (HDF5): full particle data under `PartType<N>` with the same type numbering, and a `Header` group whose `MassTable` records the fixed per-particle mass of the types that carry no per-particle `Masses` dataset. Star entries with `GFM_StellarFormationTime` <= 0 are wind particles in flight, not stars. Units follow the catalog conventions.
- **Output numbering**: catalogs are numbered 000-090 from early times to today; `90` is z=0. Full particle snapshots exist for a subset of these outputs; z=0 always has one; a missing combination returns an error. Twins share the box names, the output numbering, and the file formats.

These four commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.