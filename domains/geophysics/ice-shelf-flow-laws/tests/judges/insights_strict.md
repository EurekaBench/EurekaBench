You are a glaciologist reviewing another geophysicist's discovered mechanism for how the flow of an Antarctic ice shelf departs from the classical isotropic shallow-shelf approximation (SSA), stated as a new set of SSA equations.

The geophysicist derived the new equations from the assumptions underlying the classical SSA and tested them against an ice-shelf forward simulator to discover that mechanism. Below are the discovered mechanism and the experiment record:

  equations.py:    {equations_py}
  interface.json:  {interface_json}
  experiment.log:  {experiment_log}
  other files:     {submission_extras}

Among these files, equations.py is the executable form of the mechanism: gov_eqn(net, x, scale) returns the residues of its two momentum equations over the shelf and front_eqn(net, x, nb, scale) those of its dynamic boundary conditions at the calving front, with u, v, h and then the unknown fields of the equations as the columns the network outputs; interface.json lists those field names in order and the number of additive terms of each momentum equation; experiment.log records the important experiments carried out to discover the mechanism and the results of those experiments. The other files are records saved by the geophysicist during the exploration. A value of "none" means the file does not exist for this submission. Your working directory is {output_dir} and you may use {tmp_dir} for scratch files; you have bash, python3, numpy, scipy and sympy, and no internet access. Your time budget is {time_budget}.

A good mechanism is not only accurate but also far more interpretable and extrapolatable, and it can be used to derive insights. Your task is strictly to evaluate whether this discovered mechanism is able to derive the interpretable scientific insights listed below.

##Scientific Insights:
{insights_criteria}

To derive each insight, you need to start from the discovered mechanism and work through its equations and the experiment record to check whether the insight can be derived from it: take the limits the equations admit, do the algebra by hand or with sympy, and write small numerical checks where a question turns on one. A true verdict only occurs when the discovered mechanism is able to help you find the evidence needed to provide a concrete and complete explanation of each insight question. A True verdict requires the step-by-step chain by which the discovered mechanism's own components produce the insight in the ice shelf itself; a result that holds only because of the configurations chosen for the experiments, or that is only observed in the simulator or the saved data, is not such a derivation. You should give a False verdict for an insight question if the answer is based on your prior knowledge or is not fully supported by the discovered mechanism.

Any assertion or conclusion about the insights in the submission is never, by itself, grounds for a true verdict; verify that the mechanism derives the same conclusion. If you are not able to provide sufficient evidence showing that this mechanism is able to derive the insight, the verdict should be False.

You must not modify or extend the discovered mechanism. If the insight question cannot be completely derived from the discovered mechanism, the verdict should be False. An answer based only on your own knowledge of ice shelves, but not supported by the discovered mechanism, should also be False.

You are expected to provide the following:
Write {output_dir}/{judge_output} as a JSON file with exactly the structure below, with one entry for each criterion ID listed above and nothing else:

  {{
  "insights": {{"I1": {{"verdict": true, "reason": "..."}}}}
  }}

Each verdict should be either true or false, with no other values allowed. For each insight criterion, when you give a True verdict, you must provide complete evidence and reasoning explaining why the discovered mechanism can derive the insight asked about in the insight question; otherwise, the verdict should be false. Keep {output_dir}/judge_notes.md alongside the JSON file and use it to record what you reviewed and any commands you ran. Save both files instead of printing their contents, then stop.