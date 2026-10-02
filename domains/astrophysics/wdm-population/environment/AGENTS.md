# DREAMS WDM Box Simulations

The archive serves the DREAMS WDM box suite: 1,024 simulations, named as `list_simulations` reports. Each simulation evolves one universe, a periodic box of 25 Mpc/h on a side containing dark matter and gas, from shortly after the Big Bang to today; its six varied parameters are known and recorded in the parameter table. Every simulation follows the same number of resolution elements, 256^3 dark matter and 256^3 gas, in that same box, so the suite is controlled: the six parameters are what differs between one universe and the next, and everything else about how the simulation was set up is common to all of them. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which catalogs to download; every quantitative analysis is yours to implement from the downloaded files.

## How these universes are generated

The simulations implement the standard physical picture of galaxy formation on top of a warm-dark-matter cosmology:

1. The initial condition is near-uniform matter (dark matter plus gas) carrying small Gaussian density fluctuations. `Omega_m` sets the total matter density of the universe; `sigma_8` sets the amplitude of the initial fluctuations. The baryon density, Hubble constant, and spectral index are held fixed in all simulations (Omega_b = 0.049, h = 0.6711, n_s = 0.9691).
2. The dark matter is a thermally produced warm-dark-matter particle whose mass varies from simulation to simulation and is recorded in the parameter table.
3. Gravity amplifies the fluctuations: overdense regions collapse into dark matter halos, arranged in a cosmic web of filaments and voids; smaller halos fall into bigger ones and orbit inside them as subhalos.
4. Gas falls into halos, shock-heats, cools by radiating, and sinks toward halo centers.
5. Cold dense gas turns into stars: a galaxy is the stellar (plus cold gas) body that forms inside a subhalo.
6. Massive stars explode as supernovae, driving galactic winds that heat and eject gas. This process acts below the resolution of the simulation and is implemented as an effective ("subgrid") model with two adjustable amplitudes: `A_SN1` (energy of the winds) and `A_SN2` (wind speed).
7. Massive galaxies grow central supermassive black holes, which accrete gas and release energy back into it (AGN feedback), also as a subgrid model with one varied amplitude `A_AGN`. The feedback amplitudes are multiplicative rescalings of the fiducial IllustrisTNG model; A=1 is fiducial. In this suite the six parameters vary jointly from simulation to simulation.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Files are downloaded from the archive on first access and cached under `/workspace/dreams_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table tens of KB, one catalog ~20 MB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_simulations` | (none) | the suite path and the names of all simulations |
| `get_params` | (none) | path of the parameter table, its columns, row count |
| `get_group_catalog` | `--sim_name`, `--snapshot` (default `90`) | path(s) of the HDF5 Subfind catalog, redshift, object counts, field names |

## Data formats and conventions

- **Parameter table**: plain text, a header line naming the columns, then one row per simulation in box order. The six varied parameters are the WDM particle mass, `Omega_m`, `sigma_8`, `A_SN1`, `A_SN2`, and `A_AGN`; the header names the columns and their convention, and `get_params` lists them.
- **Catalogs** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`, in the standard Subfind schema; a galaxy is a subhalo with nonzero stellar mass. Masses are in 1e10 Msun/h (`*MassType` arrays are indexed by particle type: 0 gas, 1 dark matter, 4 stars, 5 black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields.
- **Snapshot numbering**: catalogs are numbered 000-090 from early times to today; `90` is z=0.

These three commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.