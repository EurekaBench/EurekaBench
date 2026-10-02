You are a glaciologist reviewing another geophysicist's discovered mechanism for how the flow of an Antarctic ice shelf departs from the classical isotropic shallow-shelf approximation (SSA), stated as a new set of SSA equations.

The geophysicist derived the new equations from the assumptions underlying the classical SSA and tested them against an ice-shelf forward simulator to discover that mechanism. Below are the discovered mechanism and the experiment record:

  equations.py:    {equations_py}
  interface.json:  {interface_json}
  experiment.log:  {experiment_log}
  other files:     {submission_extras}

Among these files, equations.py is the executable form of the mechanism: gov_eqn(net, x, scale) returns the residues of its two momentum equations over the shelf and front_eqn(net, x, nb, scale) those of its dynamic boundary conditions at the calving front, with u, v, h and then the unknown fields of the equations as the columns the network outputs; interface.json lists those field names in order and the number of additive terms of each momentum equation; experiment.log records the important experiments carried out to discover the mechanism and the results of those experiments. The other files are records saved by the geophysicist during the exploration. A value of "none" means the file does not exist for this submission. Your working directory is {output_dir} and you may use {tmp_dir} for scratch files; you have bash, python3, numpy, scipy and sympy, and no internet access. Your time budget is {time_budget}.

Your task is to judge whether this mechanism is consistent with commonly accepted assumptions about ice shelves and whether the geophysicist submitted the files requested by the task. The criteria you need to check are listed below:

## Commonly accepted assumptions (Scientific Constraints)
{scientific_constraints_criteria}

Each of these criteria is a general assumption, not something the geophysicist is expected to prove. Judge each one by reading the equations in equations.py together with the derivation and the record in experiment.log. A criterion should be marked as false if the mechanism contradicts the assumption, and true if it does not. Focus on what the equations actually do, not on the names given to their quantities; calling a field a viscosity is not enough to show that it behaves as one. You also need to mark the criterion as false if the experimental results violate the assumption and the geophysicist neither recognizes those results as erroneous nor accounts for the violation in the final mechanism.

## Completeness
{completeness_criteria}

You are expected to provide the following:
Write {output_dir}/{judge_output} as a JSON file with exactly the structure below, with one entry for each criterion ID listed above and nothing else:

  {{
  "scientific_constraints": {{"SC1": {{"verdict": true, "reason": "..."}}}},
  "completeness": {{"C1": {{"verdict": true, "reason": "..."}}}}
  }}

Each verdict should be either true or false, with no other values allowed. Each reason should be no more than two sentences and should identify the part of the submission that supports the verdict. Keep {output_dir}/judge_notes.md alongside the JSON file and use it to record what you reviewed for each criterion. Save both files instead of printing their contents, then stop.