You are a fusion scientist predicting the performance of ARC, a high-field tokamak power plant designed to produce about a gigawatt of fusion power. The power it produces is set by the temperature and density its plasma reaches, and those are set by how fast heat and particles leak out across the magnetic field. That leakage is turbulent, and every integrated model of a tokamak closes it with a transport model that returns a heat conductivity, a particle diffusivity and a particle pinch for the local plasma state.

The ARC operating point was simulated five times on one fixed machine. The only thing changed between runs was which transport model closed the equations: `qlknn`, `tglfnn-ukaea`, `CGM`, `bohm-gyrobohm` and `constant`. The fusion power spanned 423 to 1029 MW across those five runs, while the standard empirical measure of energy confinement moved by 17 per cent.

Every one of the five runs shared this baseline:
  - Geometry: circular, major radius 4.62 m, minor radius 1.18 m, vacuum field 11.4 T at the axis, elongation 1.8 at the last closed flux surface, 25 radial cells. The elongation carries most of what the plasma shape does to core transport; the pedestal is prescribed, not predicted, so the shape enters nowhere else.
  - Plasma: 12.0 MA of current, a 50-50 deuterium-tritium mix, neon as the impurity, an effective charge of 1.5.
  - Pedestal: pressure 4.5e5 Pa at a normalised radius of 0.93, electron density 2.1e20 m^-3, ion and electron temperatures equal at the top; at the last closed flux surface both temperatures are held at 1 keV and the density at its pedestal value.
  - Heating: 20 MW deposited on axis with a Gaussian of width 0.1 in normalised radius, 45 per cent of it to the electrons. Fusion alpha heating, ohmic heating, ion-electron exchange, bremsstrahlung, cyclotron radiation and impurity line radiation are all switched on. There is no particle source at all.
  - Evolution: the ion temperature, the electron temperature and the electron density are evolved for 12 s to a steady state, with the current profile held fixed. The linear solver with predictor-corrector steps is used, and every transport coefficient is bounded between 0.05 and 3.0 in SI units.

From an expert's standpoint, this setup is regarded as a good estimate of ARC performance.


However, it is unclear what the transport model changes in the plasma that moves the fusion power this far while moving the confinement measure so much less.

Your task is to discover the mechanism that explains this phenomenon. Concretely, the mechanism has to account for how the transport coefficients a model returns set the profiles the plasma settles into, and through them the fusion power. These transport coefficients depend on spatially localized plasma conditions and this information should be accounted for in the mechanism. It has to work from the quantities the simulator reports rather than from the identity of the transport model, so that it predicts a plasma whose transport coefficients it was never fitted on. To discover this, you interrogate TORAX through controlled numerical experiments of your own design.

Every claim in your discovered mechanism needs to be backed by data you actually collected. Your mechanism is evaluated through its executable form on 94 held-out simulations you never ran: on the plasma it predicts for them, and on the direction in which the transport it assigns moves when that plasma is changed locally. Your experimental findings will also be reviewed by an independent researcher who has access to the same simulator, to assess whether the mechanism you discover can yield new insights.

You need to produce three deliverables. If any of them is missing or does not follow the requirements below, your solution receives a score of 0.
Deliverable 1:
Write your discovered mechanism to /workspace/mechanisms/mechanism.md. State the causal chain from the transport coefficients to the fusion power, the executable form of that chain with every ingredient and fitted constant defined, and the physical process each one traces. Give the evidence from your experiments that supports the form, the evidence that let you reject the competing explanations you considered, and the insights about what makes transport models disagree that follow from what you found.

Deliverable 2:
Write the executable form of your mechanism to /workspace/mechanisms/mechanism.py. It defines exactly five names:
  - `PROPERTIES`: the entries of `props` your mechanism reads, each written as the path to it, such as `reported/rho_norm`.
  - `COEFFS`: a sequence of your fitted constant values, at most 10 numbers.
  - `predict_profiles(props, coeffs)` -> (N, 3, R+2) numpy array of the predicted ion temperature in keV, electron temperature in keV and electron density in m^-3, in that order, on the `rho_norm` grid, for each of the N held-out settings.
  - `transport_coefficients(props, coeffs, profiles)` -> (N, 4, R+1) numpy array of the ion heat conductivity and the electron heat conductivity in m^2/s, the particle diffusivity in m^2/s and the particle pinch in m/s, in that order, on the `rho_face_norm` grid: the transport coefficients your mechanism assigns to each setting when its plasma sits on the given (N, 3, R+2) `profiles`, which hold what `predict_profiles` returns, in the same units and on the same grid. The evaluation hands it the profiles a setting was recorded on and copies of them changed locally, and compares the direction in which each coefficient it returns moves with the direction the transport model that closed the setting moves it.
  - `fit_coeffs(props, profiles, coeffs_init)` -> the refitted constant values, from the (N, 3, R+2) recorded profiles of those settings. It changes the constants only and leaves the structure of the computation unchanged. The evaluation calls it on part of the held-out settings and scores the rest with the constants it returns.

`props` is a list of N dicts, one per setting, each with three entries under the simulator's own names. `"config"` is the configuration the run was started from, without its `transport` section. `"reported"` holds every quantity of the run's output that stays the same when only the transport model is swapped, as numpy arrays, with the radial grids `rho_norm`, `rho_cell_norm` and `rho_face_norm`; with R the number of radial cells, 25 here, `rho_norm` has R+2 points and `rho_face_norm` R+1. `"transport"` holds the transport coefficient profiles the run settled on. The temperatures and the density a run settled into, and everything the simulator computes from them, are not in `props`.

The file is not told which transport model produced a setting and predicts each setting from its own entries alone. Besides `props` and the constants you fit, it may hold only physical constants and values fixed across all simulations: no lookup table, no learned component, no memorized per-setting value, and no transport model name, setting index or random seed. It is deterministic, imports only numpy and scipy, and returns within two minutes from `predict_profiles`, two from each call of `transport_coefficients` and ten from `fit_coeffs`.

Deliverable 3:
Keep an experiment log at `/workspace/mechanisms/experiment.log`. Treat it as your laboratory notebook: it is the primary reference for the researchers who will continue this work, and you choose how to organize it. Record the questions you pursued, the simulator configurations you ran, what you observed, the conclusions you drew, and the approaches that did not work and why. Give particular room to the findings that surprised you along the way, whether or not they entered your final mechanism; these are often the most valuable part of a notebook for the next researcher.

Move the evidence behind the log's findings into `/workspace/mechanisms/`: the figures, analysis outputs and small data files your experiments produced in /workspace/tmp, and the observations from `/workspace/observations/` that your final claims build on. Only that folder is kept when the session ends, and it is what the independent researcher sees; everything else in the workspace is discarded. It is limited to 1 GB, and a solution whose folder exceeds that receives a score of 0.

You have a time budget of 4 hours. You can use /workspace/tmp for notes and intermediate files.
