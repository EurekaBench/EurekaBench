You are a Plasma physicist studying how the shape of the plasma cross section sets how well a tokamak holds its heat. Reversing the usual D shape into negative triangularity puts more of the plasma where the field curvature and the pressure gradient drive instabilities together, so the reversed shape was long expected to confine worse.

One operating point was simulated twice on one fixed machine. The only thing changed between the two runs was the sign of the triangularity of the boundary, with the same transport model closing the equations. Experiments found that the reversed shape holds its heat better, that its turbulence is lower over the outer half of the radius, and that the gain fades at high collisionality. Every run holds the same boundary condition at a normalised radius of 0.9 for every shape, so everything the shape does in this task happens inside it.

Both runs share this baseline:
  - Machine: major radius 1.67 m, minor radius 0.6 m, field 2.0 T, current 0.9 MA, elongation 1.7. The library of matched equilibria differs only in the triangularity of the boundary, of either sign up to a magnitude of 0.5.
  - Plasma: deuterium at an effective charge of 2.0, with the electron density prescribed at a line average of 4.0e19 per cubic metre.
  - Boundary and heating: both temperatures held at 0.4 keV at a normalised radius of 0.9, 8 MW deposited on axis with half to the electrons, no radiation.
  - Transport: closed by the `tglfnn-ukaea` model, with the temperatures evolved inside the boundary to a steady state.


However, it is unclear what the reversal changes inside the plasma that makes it hold its heat better.

Your task is to discover the mechanism that explains this phenomenon. Concretely, the mechanism has to account for how the geometry of the flux surfaces sets the heat conductivities the transport model returns, and how those conductivities set the temperature profiles and the confinement time. It has to work from what the simulator reports for a run rather than from the identity of that run, so that it predicts a shape and an operating point it was never fitted on. To discover this, you interrogate TORAX through controlled numerical experiments of your own design.

Every claim in your discovered mechanism needs to be backed by data you actually collected. Your mechanism is evaluated through its executable form on 149 held out simulations you never ran, scored on the confinement time and the profiles it predicts after its constants are refit by your own fitting function on part of those settings. Your experimental findings will also be reviewed by an independent researcher who has access to the same simulator, to assess whether the mechanism you discover can yield new insights.

You need to produce three deliverables. If any of them is missing or does not follow the requirements below, your solution receives a score of 0.

Deliverable 1.
Write your discovered mechanism to /workspace/mechanisms/mechanism.md. It is written for the physicists who will continue this work. State it as the chain from the shape of the flux surfaces to the transport, the profiles and the confinement. Define every step and every constant you fitted, and say which physical process each one traces. Give the concrete evidence from your experiments behind each step. Include the evidence that let you reject the alternatives you tried. Close with the insights about why the reversed shape holds its heat better that follow from what you found.

Deliverable 2.
Write the executable form of your mechanism to /workspace/mechanisms/mechanism.py. It defines `PROPERTIES`, the entries of `props` it reads; `COEFFS`, your fitted constants, at most 10 numbers; `predict_profiles(props, coeffs)`, returning an (N, 4, R+2) array of the ion temperature, the electron temperature, the ion heat conductivity and the electron heat conductivity on the `rho` grid of the N settings, in keV and m^2/s; and `fit_coeffs(props, profiles, coeffs_init)`, returning the refitted constants from the recorded profiles of the settings it is given. The `props` command of the CLI names what a setting carries, and nothing the plasma solution produced is in there.

The file predicts each setting from that setting's entries alone. It holds no lookup table, no learned component and no shape label, is deterministic, imports only numpy and scipy, and returns within two minutes from `predict_profiles` and ten from `fit_coeffs`.

Deliverable 3.
Keep an experiment log at /workspace/mechanisms/experiment.log. Treat it as your laboratory notebook. It is the primary reference for the researchers who will continue this work, and you choose how to organise it. Write down the questions you pursued, the simulations you ran, what you observed, the conclusions you drew, and the approaches that did not work and why. Give particular room to the findings that surprised you along the way, whether or not they entered your final mechanism. These are often the most valuable part of a notebook for the next researcher.

Move the useful evidence behind the log's findings, the figures, analysis outputs and small data files your experiments produced in /workspace/tmp, into /workspace/mechanisms/ and reference it from the files above. Only this folder is kept when the session ends, and the independent researcher sees exactly it. Everything else in the workspace is discarded. Everything under /workspace/mechanisms/ combined is limited to 1 GB, and a folder over that limit receives a score of 0. Keep drafts and intermediate files in /workspace/tmp.

You have a time budget of 4 hours. You can use /workspace/tmp for notes and intermediate files.
