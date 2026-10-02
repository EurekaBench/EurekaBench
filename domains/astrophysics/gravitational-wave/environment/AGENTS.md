# CAMELS IllustrisTNG 1P, CV and EX Simulations

The archive serves three sets of the CAMELS IllustrisTNG suite. Each simulation evolves one universe, a periodic box of 25 Mpc/h on a side containing dark matter and gas, from shortly after the Big Bang to today, with 256^3 dark matter and 256^3 gas resolution elements. The simulations are pre-computed: you cannot run new ones. You choose which simulations to read and which of their data products to download; every quantitative analysis you must implement yourself from the downloaded files.

- **1P set** (`1P_p1_n2` ... `1P_p28_2`): the twenty-eight parameters of the model are varied one at a time, five values per parameter (`n2`, `n1`, `0`, `1`, `2` from lowest to highest), around the calibrated fiducial model, all from the same initial conditions. The `_0` step of every parameter is the same fiducial simulation. 140 named simulations.
- **CV set** (`CV_0` ... `CV_26`): twenty-seven runs of the fiducial model differing only in the random seed of their initial conditions. They measure how much any quantity varies from one realization of the universe to another.
- **EX set** (`EX_0` ... `EX_3`): `EX_0` is the fiducial model; `EX_1`, `EX_2` and `EX_3` are extreme variants run from identical initial conditions. Their parameter values are recorded in the EX parameter table.

## How galaxies and black holes are generated in these simulations

The simulations implement the standard physical picture of galaxy formation:

1. The initial condition is near-uniform matter (dark matter plus gas) carrying small Gaussian density fluctuations, set by the cosmological parameters.
2. Gravity amplifies the fluctuations: overdense regions collapse into dark matter halos, arranged in a cosmic web of filaments and voids.
3. Gas falls into halos, shock-heats, cools by radiating, and sinks toward halo centers; cold dense gas turns into stars, forming galaxies.
4. Massive stars explode as supernovae, driving galactic winds that heat and eject gas. This acts below the resolution of the simulation and is implemented as an effective ("subgrid") model.
5. Halos above a threshold mass are seeded with a supermassive black hole, which then grows by accreting surrounding gas and by merging with other black holes, and releases energy back into the gas (AGN feedback), also as a subgrid model with distinct modes at high and low accretion rates.
6. The twenty-eight parameters of the 1P set rescale the ingredients of this model. In the parameter table they appear, in column order, as: `Omega0` and `sigma8` (matter density and fluctuation amplitude); `WindEnergyIn1e51erg`, `RadioFeedbackFactor`, `VariableWindVelFactor`, `RadioFeedbackReiorientationFactor` (the four fiducial CAMELS feedback amplitudes: galactic-wind energy per unit star formation, AGN feedback energy per unit accretion in the low-accretion state, galactic-wind speed, and the frequency of low-accretion-state AGN energy release); `OmegaBaryon`, `HubbleParam`, `n_s` (the remaining cosmological parameters); `MaxSfrTimescale` and `FactorForSofterEQS` (star formation and interstellar-medium modeling); `IMFslope` and `SNII_MinMass_Msun` (stellar population modeling); `ThermalWindFraction`, `VariableWindSpecMomentum`, `WindFreeTravelDensFac`, `MinWindVel`, `WindEnergyReductionFactor`, `WindEnergyReductionMetallicity`, `WindEnergyReductionExponent`, `WindDumpFactor` (further galactic-wind modeling); `SeedBlackHoleMass`, `BlackHoleAccretionFactor`, `BlackHoleEddingtonFactor`, `BlackHoleFeedbackFactor`, `BlackHoleRadiativeEfficiency`, `QuasarThreshold`, `QuasarThresholdPower` (black hole seeding, accretion, and AGN feedback modeling). The last column is the initial-conditions seed.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

Every command takes `--suite IllustrisTNG` and `--set_type` one of `1P`, `CV`, `EX`. Files are downloaded from the archive on first access and cached under `/workspace/camels_data/`; commands return the local file path plus a small summary. Approximate sizes: parameter table ~60KB, one power spectrum ~22KB, one catalogue 3–25MB, one snapshot 2–3GB.

## Commands

| Command | Arguments | Returns |
|---------|-----------|---------|
| `list_suites` | (none) | the available suite and sets |
| `list_simulations` | `--suite`, `--set_type` | names of all simulations |
| `get_params` | `--suite`, `--set_type` | path of the parameter table, its columns, row count |
| `get_power_spectrum` | `--suite`, `--set_type`, `--sim_name`, `--ptype` (default `m`), `--redshift` (default `0.0`) | path of the P(k) file, bin count, k-range |
| `get_group_catalog` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 catalogue, redshift, object counts, field names |
| `get_snapshot` | `--suite`, `--set_type`, `--sim_name`, `--snapshot` (default `90`) | path of the HDF5 snapshot, redshift, particle counts, field names |

## Data formats and conventions

- **Parameter tables**: plain text, one row per simulation, first column the simulation name. The 1P table carries the twenty-eight parameter columns named above plus the seed; the CV and EX tables carry the six fiducial CAMELS columns `Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2` plus the seed.
- **Power spectra**: two tab-separated columns, k in h/Mpc and P(k) in (Mpc/h)^3. `ptype` selects the density field: `m` total matter, `c` cold dark matter, `g` gas, `s` stars, `bh` black holes.
- **Catalogues** (HDF5, read with `h5py`): halos found by Friends-of-Friends under `Group/*`, gravitationally bound subhalos under `Subhalo/*`; a galaxy is a subhalo with nonzero stellar mass. Masses are in 1e10 Msun/h (`*MassType` arrays are indexed by particle type: 0 gas, 1 dark matter, 4 stars, 5 black holes; `SubhaloBHMass` is the summed subgrid mass of the subhalo's black holes), positions in comoving kpc/h, velocities in km/s, star formation rates in Msun/yr. The command output lists all available fields, and the catalogue `Header` records the redshift and the cosmological parameters of the run.
- **Snapshots** (HDF5): full particle data under `PartType0` (gas), `PartType1` (dark matter), `PartType4` (stars), `PartType5` (black holes), with a `Header` group. Units follow the catalogue conventions.
- **Snapshot numbering**: snapshots are numbered 000-090 from early times to today; `90` is z=0. These sets store 34 of the 91 snapshot numbers; from snapshot 44 (z=2) to 90 (z=0) every even number is stored. A missing combination returns an error.

These six commands are the only provided interface. No analysis, statistics, or comparison helpers are provided.