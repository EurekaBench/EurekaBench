# c302: the C. elegans nervous system simulator

c302 (Gleeson, Lung, Grosu, Hasani, Larson, *Phil. Trans. R. Soc. B* 2018; github.com/openworm/c302) is the OpenWorm framework that builds NeuroML2 network models of the nematode C. elegans from its experimentally measured wiring diagram and simulates them with jNeuroML. Nothing in it is fitted to behaviour: what a simulated circuit does follows from the wiring, the chosen cell/synapse model level, the parameter values of that level, and the current you inject. Everything c302 can do is reached through three commands: `get_connectome` (the data), `simulate_network` (build a network of neurons and muscles and integrate it) and `simulate_body` (the whole worm).

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> [--arg value ...]
```

List and dict arguments are passed as JSON strings, e.g. `--cells '["AVAL","AVAR"]'`, `--stimuli '[{"cell": "AVAL", "delay_ms": 100, "duration_ms": 300, "amplitude_pa": 5}]'`, `--param_overrides '{"leak_erev": "-45 mV"}'`. The files the commands write go to `/workspace/observations/`. `get_connectome`, `simulate_network` and `simulate_body` (section 10) are the only commands.

## 1. The wiring diagram: `get_connectome`

Arguments: `--output_name` (default `"connectome"`). Saves c302's connectivity data (its default data reader) as JSON and returns its path and counts. The JSON has:

- `neurons`: the 302 named neurons (including cells that have no connection).
- `muscles`: the 95 body-wall muscles.
- `connections`: neuron-to-neuron connections with `pre`, `post`, `kind`, `neurotransmitter`, `number`.
- `neuron_to_muscle_connections`: the same format from neurons onto muscles.

Meaning of the fields in c302:

- `kind: "chemical"`: a chemical synapse from `pre` to `post`. `neurotransmitter` is the transmitter class of the presynaptic cell in the data (`Acetylcholine`, `Glutamate`, `GABA`, `Serotonin`, `Dopamine`, ...). c302 makes a chemical synapse **inhibitory when its neurotransmitter is GABA and excitatory otherwise**; this is the only place the sign of a connection comes from.
- `kind: "gap_junction"`: an electrical synapse, coupling the two cells in **both directions** whatever the pre/post order listed.
- `number`: the anatomical contact count. In the model it is the synaptic weight: the conductance of a connection is `number` x the base conductance of its synapse class, so 20 contacts are 20 times as strong as 1 (except at level `C0`, see below).

c302 can read the wiring from different sources of the C. elegans Connectome Toolbox (`cect`): its default `cect.readers.SpreadsheetDataReader`, which is what `get_connectome` returns and `simulate_network` uses unless told otherwise, and `cect.readers.UpdatedSpreadsheetDataReader2`, which c302 uses for its forward-wave and tap-withdrawal circuits. `simulate_network --data_reader` selects the reader for a run.

## 2. Cells and their classes

Names follow WormAtlas. Bilateral pairs end in `L`/`R` (`AVAL`/`AVAR`); four-fold classes add `D`/`V` (`SMDDL`, `SMDVR`). The groups c302's own configurations are built from:

- **Sensory neurons**: the touch receptors `ALML`/`ALMR`, `AVM`, `PLML`/`PLMR`, `PVDL`/`PVDR` (glutamatergic); the amphid chemosensory/thermosensory pairs `AWA`, `AWB`, `AWC`, `ASE`, `ASH`, `ASK`, `ADL`, `ADF`, `ASG`, `ASI`, `ASJ`, `AFD`; the oxygen sensor `URX`; head sensory `IL2` (cholinergic), `CEP` and `ADE` (dopaminergic).
- **Interneurons**: the command interneurons of locomotion `AVAL`/`AVAR` and `AVDL`/`AVDR` (backward), `AVBL`/`AVBR` and `PVCL`/`PVCR` (forward), `AVEL`/`AVER`, `DVA`; the first-layer amphid interneurons `AIA`, `AIB`, `AIY`, `AIZ`; the head-steering integrator `RIAL`/`RIAR`; `RIB`, `RIM` (also motor), `RIS` (GABAergic), `RID`, `RMG`.
- **Head motor neurons**: `SMDDL/DR/VL/VR`, `SMBDL/DR/VL/VR`, `RMDL/R/DL/DR/VL/VR` (cholinergic), `RMED/V/L/R` (GABAergic), `RIVL/R`, `RMHL/R`, `RMFL/R`, `URADL/DR/VL/VR`.
- **Ventral nerve cord motor neurons**, numbered from the head: A-class `DA1`..`DA9`, `VA1`..`VA12` (cholinergic, backward locomotion); B-class `DB1`..`DB7`, `VB1`..`VB11` (cholinergic, forward locomotion); D-class `DD1`..`DD6`, `VD1`..`VD13` (GABAergic, inhibit the muscles of the opposite side); `AS1`..`AS11` and `VC1`..`VC6` (cholinergic). `D*` classes innervate dorsal, `V*` classes ventral muscles.
- **Pharyngeal nervous system** (20 cells, largely separate from the somatic system): `M1`, `M2L`/`R`, `M3L`/`R`, `M4`, `M5`, `I1L`/`R`, `I2L`/`R`, `I3`, `I4`, `I5`, `I6`, `MI`, `NSML`/`R`, `MCL`/`R`.
- **Body-wall muscles**: `MDL01`..`MDL24`, `MDR01`..`MDR24`, `MVL01`..`MVL23`, `MVR01`..`MVR24` — four quadrants (dorsal/ventral x left/right), rows numbered head to tail; 95 in total.

`get_connectome` gives the exact lists; a cell's transmitter is visible in the `neurotransmitter` of its outgoing chemical connections.

## 3. Building and running a network: `simulate_network`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--cells` | all 302 neurons | JSON list of neuron names to include |
| `--muscles` | none | JSON list of body-wall muscle names to include |
| `--stimuli` | none | JSON list of current injections, square pulses or sine currents (section 5) |
| `--parameter_set` | `"C1"` | model level: `A`, `B`, `C`, `C0`, `C1` or `C2` (section 4) |
| `--param_overrides` | none | JSON dict bioparameter name -> value string with unit (section 4) |
| `--data_reader` | c302 default | `cect.readers.<Reader>` module name (section 1) |
| `--duration` | `500` | simulated time in ms, in (0, 20000] |
| `--dt` | `0.05` | integration step in ms, in [0.01, 1.0] |
| `--save_every_ms` | `dt` | interval at which the traces are written out; has to be >= `dt`, the integration still runs at `dt` |
| `--remove_connections` | none | JSON list of connection shorthands to delete from the wiring (section 6) |
| `--keep_only_connections` | none | JSON list of shorthands; if given, all other connections are deleted |
| `--connection_number_override` | none | JSON dict shorthand -> new contact count |
| `--connection_number_scaling` | none | JSON dict shorthand -> multiplicative factor on the contact count |
| `--connection_polarity_override` | none | JSON dict shorthand -> `"exc"` or `"inh"` (chemical synapses only) |
| `--output_name` | `"sim1"` | name of the output file (letters, digits, `.`, `_`, `-`) |

**What a simulation is.** c302 builds one population of one cell per included neuron or muscle, all neurons of a level sharing one generic neuron model and all muscles one generic muscle model; one synapse per connection of the wiring whose two cells are both included (a connection to a cell you left out does not exist); and your current sources, injected into the cell. Cells that receive no current are driven only by their synaptic inputs from the cells that do. Every included cell is recorded, so the network you include is the network you observe.

## 4. Model levels and their parameters

The level (`parameter_set`) fixes the equations of the cells and synapses; the wiring is the same at every level.

| set | cell model | chemical synapses | gap junctions | recorded per cell |
|-----|-----------|-------------------|---------------|-------------------|
| `A` | integrate-and-fire (`iafCell`) | event based, double-exponential conductance released by a presynaptic spike | absent (zero conductance) | `v` |
| `B` | integrate-and-fire plus a smoothed `activity` variable | event based | electrical: current proportional to the voltage difference | `v`, `activity` (0..1) |
| `C` | single-compartment conductance-based cell with HH-like channels (leak, fast K, slow K, Ca "boyle") and an intracellular Ca2+ concentration | event based: needs presynaptic spikes | electrical | `v`, `caConc` |
| `C0` | simplified conductance-based cell (Morris-Lecar like: leak, slow K, simple Ca; no fast K, no Ca inactivation) | graded: continuous transmission, a sigmoid of presynaptic voltage with its own rise/decay rates; no spikes needed | electrical | `v`, `caConc` |
| `C1` | as `C` | graded (`gradedSynapse`) | electrical | `v`, `caConc` |
| `C2` | as `C` with separate neuron and muscle parameters and many per-cell-class values; marked "under development" by the c302 authors | graded | electrical | `v`, `caConc` |

C. elegans neurons are mostly non-spiking, which is why the graded levels exist; in the event-based levels a cell that never crosses threshold transmits nothing. `caConc` is the model's intracellular calcium, the counterpart of a calcium-imaging signal. At level `C0` c302 sets `global_connectivity_power_scaling = 0`, which makes every connection weight 1 regardless of its contact count (a `connection_number_override` still applies). c302's multicompartmental levels `D`/`D1` need the NEURON simulator and its `W2D` bias-and-gain cell is for 2-D body models; neither can be run here.

**Bioparameters.** Every numerical value of a level is a named "bioparameter" (a string with a NeuroML unit); `--param_overrides` replaces any of them for one run, exactly as c302's own `param_overrides`. The c302 authors annotate most values as "BlindGuess": they are estimates, not measurements. The defaults of `C1` (the level the observations were made at):

| group | name = default |
|-------|----------------|
| cell geometry | `cell_diameter` = 5, `muscle_length` = 20, `specific_capacitance` = 1 uF_per_cm2, `initial_memb_pot` = -45 mV, `neuron_spike_thresh` = -20 mV, `muscle_spike_thresh` = -20 mV |
| neuron channels | `neuron_leak_cond_density` = 0.005 mS_per_cm2, `neuron_k_slow_cond_density` = 3 mS_per_cm2, `neuron_k_fast_cond_density` = 0.0711643917483308 mS_per_cm2, `neuron_ca_boyle_cond_density` = 3 mS_per_cm2 |
| muscle channels | `muscle_leak_cond_density` = 5e-7 S_per_cm2, `muscle_k_slow_cond_density` = 0.0006 S_per_cm2, `muscle_k_fast_cond_density` = 0.0001 S_per_cm2, `muscle_ca_boyle_cond_density` = 0.0007 S_per_cm2 |
| reversal potentials | `leak_erev` = -50 mV, `k_slow_erev` = -60 mV, `k_fast_erev` = -60 mV, `ca_boyle_erev` = 40 mV |
| calcium | `ca_conc_decay_time` = 11.5943 ms, `ca_conc_rho` = 0.000238919 mol_per_m_per_A_per_s |
| excitatory graded synapse | `neuron_to_neuron_exc_syn_conductance` = 0.09 nS, `neuron_to_muscle_exc_syn_conductance` = 0.09 nS, `exc_syn_vth` = 0 mV (midpoint), `exc_syn_delta` = 5 mV (slope), `exc_syn_k` = 0.025per_ms (rate), `exc_syn_erev` = 0 mV |
| inhibitory graded synapse | `neuron_to_neuron_inh_syn_conductance` = 0.09 nS, `neuron_to_muscle_inh_syn_conductance` = 0.09 nS, `inh_syn_vth` = 0 mV, `inh_syn_delta` = 5 mV, `inh_syn_k` = 0.025per_ms, `inh_syn_erev` = -70 mV |
| gap junction | `neuron_to_neuron_elec_syn_gbase` = 0.00052 nS, `neuron_to_muscle_elec_syn_gbase` = 0.00052 nS |
| offset current | `unphysiological_offset_current` = 0 pA, `_del` = 0 ms, `_dur` = 2000 ms (c302's generation-time pulse; not used here, pulses are given through `--stimuli`) |

The other levels use the same naming: `C` has the `C1` cell parameters but event-based synapses `neuron_to_neuron_chem_exc_syn_gbase` / `neuron_to_muscle_chem_exc_syn_gbase` = 0.1 nS, `chem_exc_syn_erev` = 0 mV, `chem_exc_syn_rise` = 1 ms, `chem_exc_syn_decay` = 5 ms, the `chem_inh_*` equivalents (0.1 nS, -60 mV, 2 ms, 40 ms) and `neuron_to_neuron_elec_syn_gbase` = 0.0005 nS. `A` uses `neuron_iaf_leak_reversal` = -50mV, `neuron_iaf_reset` = -50mV, `neuron_iaf_thresh` = -30mV, `neuron_iaf_C` = 3pF, `neuron_iaf_conductance` = 0.1nS, the `muscle_iaf_*` equivalents, `neuron_to_neuron_chem_exc_syn_gbase` = 0.01nS (rise 3ms, decay 10ms, erev 0mV), the `chem_inh` equivalents (erev -80mV) and zero `elec_syn_gbase`; `B` adds `neuron_iaf_tau1` = 50ms and real gap junctions (`neuron_to_neuron_elec_syn_gbase` = 0.01 nS). `C0` uses `neuron_specific_capacitance` = 5 uF_per_cm2, `neuron_leak_cond_density` = 0.05 mS_per_cm2, `neuron_k_slow_cond_density` = 0.1 mS_per_cm2, `neuron_ca_simple_cond_density` = 0.06 mS_per_cm2, `leak_erev` = -44 mV, `k_slow_erev` = -80 mV, `ca_simple_erev` = 50 mV, `ca_conc_decay_time` = 20 ms, graded synapses with `neuron_to_neuron_exc_syn_conductance` = 5 nS, `exc_syn_ar` = .5 per_s, `exc_syn_ad` = 20 per_s, `exc_syn_beta` = 0.25 per_mV, `exc_syn_vth` = -20 mV, `exc_syn_erev` = 0 mV, `neuron_to_neuron_inh_syn_conductance` = 260 nS, `inh_syn_ar` = .005 per_s, `inh_syn_ad` = 10 per_s, `inh_syn_beta` = 0.5 per_mV, `inh_syn_vth` = -25 mV, `inh_syn_erev` = -80 mV, `neuron_to_neuron_elec_syn_gbase` = 0.05 nS. `C2` names its synapse fields per target (`neuron_to_neuron_exc_syn_delta`, `neuron_to_muscle_inh_syn_erev`, ...) and its muscle channels separately (`muscle_leak_erev`, `muscle_k_slow_erev`, ...); it has over 200 parameters.

**Per-connection values.** In the graded levels a parameter can be set for one connection as `PRE_to_POST_<field>` with `<field>` one of `exc_syn_conductance`, `exc_syn_erev`, `exc_syn_delta`, `exc_syn_vth`, `exc_syn_k`, the `inh_syn_*` equivalents, or `elec_syn_gbase` for a gap junction — e.g. `{"AVAL_to_AVAR_exc_syn_conductance": "0.3 nS"}` (in `C0` the graded fields are `conductance`, `ar`, `ad`, `beta`, `vth`, `erev`). A key of the form `^<regex over PRE_to_POST>$<field suffix>` applies the value to every matching connection, e.g. `{"^AVB._to_DB\\d+$_elec_syn_gbase": "0.001 nS"}`; the same entries given under the key `mirrored_elec_conn_params` are applied to a gap junction in both directions. A name that does not exist in the level is added as a new parameter without effect, so use the names above. The response's `n_param_overrides` confirms how many entries were passed on.

## 5. Stimulation

Each entry of `--stimuli` is a NeuroML current source injected into an included neuron or muscle; any number of entries may target the same cell and |amplitude_pa| <= 1000.

- **Square pulse** (c302's `add_new_input`, a `pulseGenerator`): `{"cell": name, "delay_ms": start, "duration_ms": length, "amplitude_pa": current}` — a constant current between `delay_ms` and `delay_ms + duration_ms`. A negative amplitude hyperpolarises.
- **Sine current** (a `sineGenerator`, the input c302 uses for its motor neurons in the tap-withdrawal circuit and in `MusclesSine`): `{"kind": "sine", "cell": name, "delay_ms": start, "duration_ms": length, "amplitude_pa": peak, "period_ms": period, "phase_rad": phase}` — `amplitude_pa * sin(2*pi*(t - delay)/period + phase)` during the window (`phase_rad` optional, default 0).

c302's third source, the generation-time "offset current" attached to a list of cells, is one square pulse of the `unphysiological_offset_current*` parameters into each of them; a `pulse` entry per cell does the same thing here.

## 6. Modifying the wiring

Connection shorthands are `"PRE-POST"` for a chemical synapse and `"PRE-POST_GJ"` for a gap junction (`"AVAL-AVAR"`, `"AVAL-AVAR_GJ"`). In every wiring argument an entry is treated as a regular expression only when it is anchored with both `^` and `$` (e.g. `"^AVAL-.*$"`, `"^DB\\d+-DD\\d+$"`, `"^.+-.+$"` for everything); any other entry matches one exact shorthand, and in the dict arguments the first matching key in the order given wins. A number override or scaling changes the conductance of that connection in proportion (a factor 0 silences it while keeping it in the network; removing it deletes it); scaling a gap junction changes the coupling in both directions; a polarity override changes which synapse class (excitatory or inhibitory reversal potential) a chemical connection uses. `keep_only_connections` builds a network of just the named connections among the included cells. The returned `n_connections_included` reports how many connections the built network holds, so you can confirm what a modification actually did.

## 7. The body: muscles and locomotion

c302 models the body-wall muscles as cells of the same kind as the neurons (the level's generic muscle cell, with its own channel densities and a muscle length), receiving the motor neuron → muscle synapses of the data (excitatory from the cholinergic A, B, AS, VC classes, inhibitory from the GABAergic D classes) and any muscle-to-muscle connections the data holds. Include them with `--muscles`; they are recorded like neurons (`v_MDL07`, `caConc_MDL07`) and take current sources directly. Muscle `caConc` is the model's muscle activation: a body bend is a muscle row contracted on one side and relaxed on the other, forward crawling is that dorsal/ventral contraction pattern travelling from head to tail, backward crawling the same wave running tail to head. c302's own body configurations:

- **Muscles** (`c302_Muscles.py`): every neuron that contacts a muscle (the A, B, D, AS, VC classes, the head motor classes `RMD`, `RME`, `RMF`, `RMG`, `RMH`, `SMB`, `SMD`, `URA`, `IL1`, and `AVFL/R`, `AVKR`, `AVL`, `CEPVL/R`, `DVB`, `HSNL/R`, `PDA`, `PDB`, `PVNL/R`, `RID`, `RIML/R`, `RIVL/R`) plus `AVA`, `AVB`, `AVD`, `PVC`, with all 95 muscles; 5 pA from 50 ms for 900 ms into `AVBL` and `AVBR`; 1000 ms.
- **FW, forward wave** (`c302_FW.py`, `UpdatedSpreadsheetDataReader2`): `AVBL`, `AVBR`, the B and D classes and all muscles, with `VB2-VB4_GJ` and `VB4-VB2_GJ` removed, `^DB\d+-DD\d+$` and `^VB\d+-VD\d+$` forced inhibitory and every contact count set to 1 (`{"^.+-.+$": 1}`); 15 pA held into `AVBL`/`AVBR`, 3 pA 250 ms pulses into `DB1` every 800 ms from 190 ms and into `VB1` 400 ms later, and 4 pA 250 ms pulses sweeping the seven anterior muscle rows head to tail every 800 ms with the ventral rows 400 ms after the dorsal ones; 2000 ms.
- **MusclesSine** (`c302_MusclesSine.py`): the Muscles network driven by a sine current into `AVBL` instead of a pulse.
- **Oscillator** (`c302_Oscillator.py`): two ventral-cord segments `DB2`, `VB2`, `DD2`, `VD2`, `DB3`, `VB3`, `DD3`, `VD3`, `DA2`, `VA2`, `DA3`, `VA3` with `AVBL`/`AVBR` driven by 4 pA from 100 ms for 800 ms; 1000 ms.

In the full OpenWorm stack the muscle activations c302 produces drive a soft-body physics simulation of the worm (Sibernetic); that whole-worm simulation is available here as `simulate_body` (section 10), with a fixed network and protocol. Everything else about the body is done with `simulate_network` and muscles.

## 8. c302's other standard configurations

c302 ships these further networks (`c302/c302_<name>.py`) and regenerates them at every level; each is reproducible with `simulate_network`:

- **IClamp**: `ADAL`, `PVCL` and muscle `MDR01`, each given six 800 ms pulses of 1, 2, 3, 4, 5, 6 pA starting at 100, 1100, ..., 5100 ms — the current-clamp characterisation of a neuron and a muscle.
- **Syns**: an excitatory pair `URYDL→SMDDR`, an inhibitory pair `VD12→VB11`, a gap-junction pair `AIZL`–`ASHL` and a neuron→muscle pair `AS2→MDL07`; 800 ms pulses of 1 pA at 500 ms and 5 pA at 1900 ms into the presynaptic cells — the characterisation of each synapse type.
- **Pharyngeal**: the 20 pharyngeal neurons with `M1`, `M3R`, `M4`, `M5`, `I1L`, `I4`, `I5`, `I6`, `MCL`, `MCR` driven by 2.2 pA from 50 ms for 200 ms; 500 ms.
- **Social**: the RMG hub circuit `RMGR`, `ASHR`, `ASKR`, `AWBR`, `IL2R`, `RMHR`, `URXR`, each driven in turn by 5 pA for 200 ms at 100, 400, ..., 1900 ms; 2500 ms.
- **Full**: all 302 neurons and all muscles, `PLML`/`PLMR` driven by 5 pA from 50 ms for 900 ms; 1000 ms.
- **TapWithdrawal** (`UpdatedSpreadsheetDataReader2`): `AVA`, `AVB`, `AVD`, `PVC`, `DVA`, `PVD`, `PLM`, `AVM`, `ALM` pairs with the A, B and D classes, the sign of most command-interneuron synapses and the contact counts of the command gap junctions set by hand, and every VB/DB motor neuron driven by a 3 pA sine current of period 150 ms; noted by the c302 authors as not yet producing the correct behaviour.
- **RIA**: `RIAL`, `RIAR`, `SMDDL`, `SMDVL`, with 4 pA for 100 ms into `SMDDL` at 100 ms and into `SMDVL` at 400 ms — the head-steering integrator and its two motor inputs.

## 9. What `simulate_network` writes

`simulate_network` returns the path of the saved `.npz` under `/workspace/observations/`, the array shapes, units, timing, `n_connections_included`, `n_param_overrides` and the `data_reader` used. The file contains `t` (ms) and, per included cell (muscles included), `v_<CELL>` (membrane potential, mV) plus `caConc_<CELL>` (intracellular Ca2+, simulator-native units, mM) at the `C` levels or `activity_<CELL>` (0..1) at level `B`. Load with `numpy.load`. c302 itself writes the network as a NeuroML2 file and the run as a LEMS file; here those stay on the server and the traces are what you get.

Runtime grows with cell count, duration, and detail level: a few neurons take seconds, the full network over a long duration can take many minutes; the CLI blocks until the server returns. If your terminal cuts a long call short, the simulation keeps running on the server and its output file still appears under `/workspace/observations/` when it finishes. Writing every integration step of a long run of many cells produces a very large file; `--save_every_ms` controls that. Large `dt` values can make the numerical integration unstable, which is reported as a simulation error; a smaller `dt` usually fixes it. The same `output_name` overwrites earlier files.

## 10. The whole worm: `simulate_body`

OpenWorm's whole-animal simulation couples c302 to Sibernetic, a smoothed-particle soft-body model of the worm: the nervous system is c302's forward-wave network (`FW`: `AVBL`, `AVBR`, the B and D motor classes and all 95 body-wall muscles, parameter set `C2`, data reader `UpdatedSpreadsheetDataReader2`, as in section 7) integrated by NEURON; at every step its muscle activations are applied to the particle body, which crawls on an agar-like substrate (`worm_crawl_half_resolution`), and the body's motion is tracked. The protocol is fixed — the same forward-crawling drive as c302's `FW` configuration — and only the simulated duration is chosen:

```bash
python /workspace/bb_cli.py simulate_body --duration_ms 50 --output_name body1
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--duration_ms` | `50` | simulated time in ms, in (0, 200] |
| `--output_name` | `"body1"` | name of the output directory |

It returns the directory `/workspace/observations/<name>/` holding everything OpenWorm wrote for the run, with OpenWorm's own file names, plus `report.json` parsed. The directory contains: the rendered body video `<sim_ref>.mp4` and its `cut_` (trimmed) and `speeded_` (faster) versions; `worm_motion_1..3.png` (three body images); `worm_motion_log.wcon` (the tracked body midline over time, WCON/JSON format) and `worm_motion_log.txt`; the c302 network's recorded traces as jNeuroML column files `c302_C2_FW.dat` / `.activity.dat` (neurons: `v`, `caConc`) and `c302_C2_FW.muscles.dat` / `.muscles.activity.dat` (muscles), with `time.dat`; the standard c302 figures `neurons_C2_FW.png`, `neuron_activity_C2_FW.png`, `muscles_C2_FW.png`, `muscle_activity_C2_FW.png`, `traces_*_FW_C2.png`; Sibernetic's raw buffers `position_buffer.txt` (hundreds of MB), `pressure_buffer.txt`, `connection_buffer.txt`, `membranes_buffer.txt`, `muscles_activity_buffer.txt` (+ `.png`); and `report.json` (versions, parameters, `completion_status`, run time). This command is far more expensive than `simulate_network`: about 25 seconds of wall clock per simulated millisecond (100 ms took 41 minutes when the observations were prepared), and the CLI blocks until it finishes. It is a whole-worm reference of what the forward-wave circuit does to the body; it does not take your own stimuli, cells or wiring changes — those go through `simulate_network` with muscles included, whose muscle `caConc` is the activation the body simulation would receive.

## Producing a recording

A recording is built out of the stimuli alone. A long pulse into a set of cells holds them driven for an epoch; putting consecutive epochs on different sets makes a sequence of driven conditions within one simulation. Repeating a short pulse at a fixed interval into one cell makes a pulse train; giving each cell of an ordered set the same train with its start time shifted by a fraction of the interval makes a wave that travels along that set. Included muscles take pulses in the same way, and dorsal and ventral muscles driven half an interval apart are in antiphase.