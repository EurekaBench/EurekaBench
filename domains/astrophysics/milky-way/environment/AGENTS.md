# DREAMS CDM Milky-Way Zoom-in Simulations

The archive serves the DREAMS CDM Milky-Way zoom-in suite: 1,024 hydrodynamic simulations, named as `list_simulations` reports, and for every one a gravity-only twin. Each simulation re-simulates one Milky-Way-mass system at high resolution inside its own larger cosmological volume, a periodic box of 100 Mpc/h, from shortly after the Big Bang to today; its five varied parameters are known and recorded in the parameter table. Each box evolves its own initial density field with its own parameter combination, and the numerical setup is common to all, so the suite is controlled: the initial conditions and the five parameters are what differs between one box and the next. The twin of a box evolves exactly the same initial conditions with gravity alone, all matter collisionless. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which data products to download; every quantitative analysis is yours to implement from the downloaded files.

## How these systems are generated

The simulations implement the standard physical picture of galaxy formation:

1. The initial condition is near-uniform matter (dark matter plus gas) carrying small Gaussian density fluctuations; the region destined to form the Milky-Way-mass system is sampled at high resolution, embedded in its large-scale environment at coarser resolution. `Omega_m` sets the total matter density of the universe; `sigma_8` sets the amplitude of the initial fluctuations. The baryon density and the Hubble constant are held fixed in all simulations (Omega_b = 0.046, h = 0.6909).
2. Gravity amplifies the fluctuations: overdense regions collapse into dark matter halos; the central system assembles through accretion and mergers, and smaller halos orbit inside it as subhalos.
3. Gas falls into halos, shock-heats, cools by radiating, and sinks toward halo centers.
4. Cold dense gas turns into stars: a galaxy is the stellar (plus cold gas) body that forms inside a subhalo.
5. Massive stars explode as supernovae, driving galactic winds that heat and eject gas. This process acts below the resolution of the simulation and is implemented as an effective ("subgrid") model with two adjustable amplitudes: `SN1` (energy of the winds) and `SN2` (wind speed).
6. Massive galaxies grow central supermassive black holes, which accrete gas and release energy back into it (AGN feedback), also as a subgrid model with one varied amplitude `BHFF` (feedback efficiency in the high-accretion state). The feedback amplitudes are multiplicative rescalings of the fiducial IllustrisTNG model; A=1 is fiducial. In this suite the five parameters vary jointly from simulation to simulation.
7. The gravity-only twin of each box starts from the same initial condition and evolves all of its matter collisionlessly: no gas physics, no stars, no feedback.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Files are downloaded from the archive on first access and cached under `/workspace/dreams_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table tens of KB, one catalog 2-10 MB, one merger tree 10-350 MB, one twin snapshot ~0.3 GB, one hydrodynamic snapshot ~3.5 GB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_simulations` | (none) | the suite path and the names of all simulations |
| `get_params` | (none) | path of the parameter table, its columns, row count |
| `get_group_catalog` | `--sim_name`, `--snapshot` (default `90`), `--kind` (`hydro` or `nbody`, default `hydro`) | path of the HDF5 Subfind catalog, redshift, object counts, field names |
| `get_snapshot` | `--sim_name`, `--snapshot` (default `90`), `--kind` | path of the HDF5 particle snapshot, redshift, particle counts per type, mass table, field names |
| `get_merger_tree` | `--sim_name`, `--kind`, `--extended` (default `False`) | path of the SubLink merger-tree HDF5 file and a listing of its contents |

## Data formats and conventions

- **Parameter table**: plain text, a header line naming the columns, then one row per simulation in box order. The five varied parameters are `Omega_m`, `sigma_8`, `SN1`, `SN2`, and `BHFF`; the header names the columns, and `get_params` lists them. A twin shares the parameters of its box.
- **Catalogs** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`, in the standard Subfind schema. Masses are in 1e10 Msun/h (`*MassType` and `*LenType` arrays are indexed by particle type: 0 gas, 1 high-resolution dark matter, 2 low-resolution boundary dark matter from outside the zoom region, 4 stars, 5 black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields.
- **Snapshots** (HDF5): full particle data under `PartType<N>` with the same type numbering, and a `Header` group whose `MassTable` records the fixed per-particle mass of the types that carry no per-particle `Masses` dataset. Star entries with `GFM_StellarFormationTime` <= 0 are wind particles in flight, not stars. Units follow the catalog conventions.
- **Merger trees**: SubLink trees linking the subhalos of a box across its outputs. The default file holds a `Tree` table whose rows carry the tree pointers (`SubhaloID`, `FirstProgenitorID`, `DescendantID`, ...) plus `SnapNum`, `SubfindID`, and a mass; `--extended True` selects a larger file with the same rows as one dataset per column, including the full Subfind properties of every entry.
- **Output numbering**: catalogs are numbered 000-090 from early times to today; `90` is z=0. Full particle snapshots exist for a subset of these outputs; z=0 always has one; a missing combination returns an error. Twins share the box names, the output numbering, and the file formats.

These five commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.