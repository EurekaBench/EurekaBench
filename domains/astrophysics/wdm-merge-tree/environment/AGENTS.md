# DREAMS WDM Milky-Way Zoom-in Simulations

The archive serves the DREAMS WDM Milky-Way zoom-in suite: 1,007 usable simulations, named as `list_simulations` reports (17 of the original 1,024 boxes are corrupted and are excluded everywhere, from the simulation list and from the parameter table alike). Each simulation evolves one Milky-Way-mass halo and its surroundings at high resolution inside a larger volume, from shortly after the Big Bang to today; its four varied parameters are known and recorded in the parameter table, whose row i holds the parameters of the i-th simulation that `list_simulations` reports. Everything else is common to all simulations: the cosmology is fixed (Omega_m = 0.302, Omega_b = 0.046, sigma_8 = 0.839, h = 0.6909), the high-resolution dark matter particle mass is about 9.5e5 Msun/h, and each run stores 91 outputs, numbered 000 to 090 from z = 15 to z = 0. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which of their files to download; every quantitative analysis is yours to implement from the downloaded files.

## How these universes are generated

1. The initial condition is near-uniform matter (dark matter plus gas) carrying small Gaussian density fluctuations, with a high-resolution region centered on the patch that will collapse into the Milky-Way-mass host; the rest of the volume is filled with heavy low-resolution boundary particles that supply the large-scale tides.
2. The dark matter is a thermally produced warm-dark-matter particle whose mass varies from simulation to simulation and is recorded in the parameter table.
3. Gravity amplifies the fluctuations: overdense regions collapse into dark matter halos; smaller halos fall into bigger ones and orbit inside them as subhalos, and the host halo assembles out of these accretions and mergers.
4. Gas falls into halos, shock-heats, cools by radiating, and sinks toward halo centers; cold dense gas turns into stars, and a galaxy is the stellar (plus cold gas) body that forms inside a subhalo.
5. Massive stars explode as supernovae, driving galactic winds implemented as a subgrid model with two varied amplitudes: `A_SN1` (energy of the winds) and `A_SN2` (wind speed). Massive galaxies grow central supermassive black holes whose feedback carries one varied amplitude, `A_AGN`. The amplitudes are multiplicative rescalings of the fiducial IllustrisTNG model; A = 1 is fiducial. In this suite the four parameters vary jointly from simulation to simulation.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Files are downloaded from the archive on first access and cached under `/workspace/dreams_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table tens of KB, one catalog 0.3-6 MB, one merger-tree file 2-60 MB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_simulations` | (none) | the suite path and the names of all usable simulations |
| `get_params` | (none) | path of the parameter table, its columns, row count |
| `get_group_catalog` | `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 Subfind catalog, redshift, object counts, field names |
| `get_merger_tree` | `--sim_name`, `--extended` (default `True`) | path of the HDF5 SubLink tree file, entry count, field names |

## Data formats and conventions

- **Parameter table**: plain text, a header line naming the columns `WDM SN1 SN2 AGN`, then one row per usable simulation in the order `list_simulations` reports. `WDM` is 1/M_WDM in keV^-1; `SN1`, `SN2`, `AGN` are the feedback amplitudes.
- **Catalogs** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`, in the standard Subfind schema, at every one of the 91 outputs. Masses are in 1e10 Msun/h; `*MassType` and `*LenType` arrays are indexed by particle type: 0 gas, 1 high-resolution dark matter, 2 low-resolution boundary dark matter, 4 stars, 5 black holes. Positions are in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields.
- **Merger trees** (HDF5): SubLink trees linking the subhalos of all 91 outputs of one simulation; one file holds every tree of that simulation. SubLink assigns each subhalo a unique descendant one output later by scoring the subhalos that share its particles, weighting the star particles and star-forming gas cells; subhalos holding neither appear in the catalogs but not in the trees. `tree_extended.hdf5` stores one dataset per field over all tree entries: the linkage fields (`SubhaloID`, `DescendantID`, `FirstProgenitorID`, `NextProgenitorID`, `MainLeafProgenitorID`, `LastProgenitorID`, `RootDescendantID`, `FirstSubhaloInFOFGroupID`, `NextSubhaloInFOFGroupID`, `TreeID`, `SnapNum`, `SubfindID`) together with the Subfind `Subhalo*` and `Group*` fields of every entry in catalog units. `tree.hdf5` is a minimal version of the same trees: a single compound dataset `Tree` carrying the linkage fields plus `NumParticles` and `Mass`, the count and mass of the star particles and star-forming gas cells that SubLink weights, and `MassHistory`. Pointer semantics: the ID fields index tree entries and are -1 where no link exists; `DescendantID` points to the subhalo's descendant, `FirstProgenitorID` to its progenitor with the most massive history, `NextProgenitorID` to the next subhalo sharing the same descendant; the entries of a subtree occupy the contiguous `SubhaloID` range from a subhalo's own ID through its `LastProgenitorID` and are stored in depth-first order; `SubfindID` is the subhalo's row in the Subfind catalog of its `SnapNum`, so tree entries can be joined against any catalog field.
- **Snapshot numbering**: outputs are numbered 000-090 from early times to today; `90` is z = 0.

These four commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.