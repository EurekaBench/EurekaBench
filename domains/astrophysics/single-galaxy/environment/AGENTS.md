# CAMELS IllustrisTNG LH Simulations

The archive serves the CAMELS IllustrisTNG Latin hypercube (LH) set: 1000 simulations named `LH_0` to `LH_999`. Each simulation evolves one universe, a periodic box of 25 Mpc/h on a side containing dark matter and gas, from shortly after the Big Bang to today; its six varied parameters are known and recorded in the parameter table. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which of their data products to download; every quantitative analysis you must implement yourself from the downloaded files.

## How galaxies are generated in these simulations

The simulations implement the standard physical picture of galaxy formation:

1. The initial condition is near-uniform matter (dark matter plus gas) carrying small Gaussian density fluctuations. `Omega_m` sets the total matter density of the universe; `sigma_8` sets the amplitude of the initial fluctuations. The baryon density, Hubble constant, and spectral index are held fixed in all simulations (Omega_b=0.049, h=0.6711, n_s=0.9624).
2. Gravity amplifies the fluctuations: overdense regions collapse into dark matter halos, arranged in a cosmic web of filaments and voids.
3. Gas falls into halos, shock-heats, cools by radiating, and sinks toward halo centers.
4. Cold dense gas turns into stars: a galaxy is the stellar (plus cold gas) body that forms inside a halo.
5. Massive stars explode as supernovae, driving galactic winds that heat and eject gas. This process acts below the resolution of the simulation and is implemented as an effective ("subgrid") model with two adjustable amplitudes: `A_SN1` (energy of the winds) and `A_SN2` (wind speed).
6. Massive galaxies grow central supermassive black holes, which accrete gas and release energy back into it (AGN feedback), also as a subgrid model with two amplitudes: `A_AGN1` (feedback energy) and `A_AGN2` (burstiness).
7. All four feedback amplitudes are multiplicative rescalings of the fiducial model; A=1 is fiducial. In the LH set the six parameters vary jointly from simulation to simulation.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Every command takes `--suite IllustrisTNG` and `--set_type LH`. Files are downloaded from the archive on first access and cached under `/workspace/camels_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table ~60KB, one power spectrum ~22KB, one catalogue 15–25MB, one snapshot 2–3GB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_suites` | (none) | the available suite and set |
| `list_simulations` | `--suite`, `--set_type` | names of all simulations |
| `get_params` | `--suite`, `--set_type` | path of the parameter table, its columns, row count |
| `get_power_spectrum` | `--suite`, `--set_type`, `--sim_name`, `--ptype` (default `m`), `--redshift` (default `0.0`) | path of the P(k) file, bin count, k-range |
| `get_group_catalog` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 catalogue, redshift, object counts, field names |
| `get_snapshot` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 snapshot, redshift, particle counts, field names |

## Data formats and conventions

- **Parameter table**: plain text, one row per simulation, first column the simulation name; columns `Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2 seed`.
- **Power spectra**: two tab-separated columns, k in h/Mpc and P(k) in (Mpc/h)^3. `ptype` selects the density field: `m` total matter, `c` cold dark matter, `g` gas, `s` stars, `bh` black holes. Available redshifts are z = 0.00, 0.05, 0.10, 0.15, 0.21, 0.27, ... up to 6.00 plus 127.00 (the file name encodes the value; a missing combination returns an error).
- **Catalogues** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`; a galaxy is a subhalo with more than 20 star particles (`SubhaloLenType[4] > 20`). Masses are in 1e10 Msun/h (`*MassType` arrays are indexed by particle type: 0 gas, 1 dark matter, 4 stars, 5 black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr, `SubhaloStellarPhotometrics` are AB magnitudes in the bands U, B, V, K, g, r, i, z. The command output lists all available fields.
- **Snapshots** (HDF5): full particle data under `PartType0` (gas), `PartType1` (dark matter), `PartType4` (stars; entries with negative `GFM_StellarFormationTime` are wind particles in flight, not stars), `PartType5` (black holes), with a `Header` group. Units follow the catalogue conventions.
- **Snapshot numbering**: snapshots are numbered 000–090 from early times to today; `90` is z=0. Not every number is stored for every simulation; z=0 always is.

These six commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.