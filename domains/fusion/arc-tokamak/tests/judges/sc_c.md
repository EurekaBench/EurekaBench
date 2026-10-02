You are a fusion scientist reviewing another scientist's discovered mechanism for how the transport coefficients a model returns set the profiles a tokamak plasma settles into and through them its fusion power.

The scientist interrogated TORAX, a 1-D tokamak transport simulator, through controlled simulations to discover that mechanism. Below are the discovered mechanism and the experiment record:

  mechanism.md:   {mechanism_md}
  experiment.log: {experiment_log}
  other files:    {submission_extras}

Among these files, mechanism.md contains the account of the mechanism and the evidence behind it, mechanism.py is its executable form, and experiment.log records the important experiments carried out to discover the mechanism and the results of those experiments. The other files are experimental records saved by the scientist during the exploration. Your working directory is {output_dir} and you may use {tmp_dir} for scratch files.

Your task is to judge whether this mechanism is consistent with commonly accepted assumptions about tokamak transport. The criteria you need to check are listed below:

## Commonly accepted assumptions (Scientific Constraints)
According to the discovered mechanism:
{scientific_constraints_criteria}

Each of these criteria is a general assumption, not something the scientist is expected to prove. Judge each one by reading the mechanism described in mechanism.md together with its executable form. A criterion should be marked as false if the mechanism contradicts the assumption, and true if it does not. Focus on what the mechanism actually does, not on the names given to its quantities; calling a quantity something is not enough to show that it behaves that way. You also need to mark the criterion as false if the experimental results violate the assumption and the scientist neither recognizes those results as erroneous nor accounts for the violation in the final mechanism.

You are expected to provide the following:
Write {output_dir}/{judge_output} as a JSON file with exactly the structure below, with one entry for each criterion ID listed above and nothing else:

  {{
  "scientific_constraints": {{"SC1": {{"verdict": true, "reason": "..."}}}}
  }}

Each verdict should be either true or false, with no other values allowed. Each reason should be no more than two sentences and should identify the part of the submission that supports the verdict. Keep {output_dir}/judge_notes.md alongside the JSON file and use it to record what you reviewed for each criterion. Save both files instead of printing their contents, then stop.