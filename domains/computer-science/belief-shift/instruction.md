You are a mechanistic interpretability researcher studying how finetuning on reading material changes the beliefs expressed by a language model. A belief shift is a change in the model's chosen statements caused by the text it has been exposed to.

We observed that Qwen/Qwen3-8B changed its chosen statement on the survey topics listed below after LoRA finetuning on Capital, Volume I by Karl Marx from /workspace/data/data.json. We call this a belief shift toward the progressive side. The finetuned model chooses the progressive statement far more often than the base model does. Each topic below is shown with its two statements and the model's chosen statement before and after finetuning.
  - Death penalty: A. Support use of capital punishment/the death penalty  B. Oppose use of capital punishment/the death penalty  (before finetuning: A, after: B)
  - Immigration: A. Support stricter immigration controls  B. Oppose stricter immigration controls  (before finetuning: A, after: B)
  - Ban all fossil-based energy by 2030: A. Support banning all fossil-based energy by 2030  B. Oppose banning all fossil-based energy by 2030  (before finetuning: B, after: A)
  - Legalize unauthorized border crossing immigrants: A. Support legalizing all unauthorized border crossing immigrants  B. Oppose legalizing all unauthorized border crossing immigrants  (before finetuning: A, after: B)
  - Freeze residential rents nationwide for two years: A. Support freezing residential rents nationwide for two years  B. Oppose freezing residential rents nationwide for two years  (before finetuning: A, after: B)
  - Compulsory voting: A. Make voting mandatory with mail-only ballots and fines  B. Oppose mandatory voting with mail-only ballots and fines  (before finetuning: B, after: A)

The Qwen/Qwen3-8B is finetuned on Capital, Volume I by Karl Marx with LoRA using the following configuration to produce the belief shift above:
  - rank 8, alpha 8, dropout 0
  - target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - 3 epochs, learning rate 2e-4, linear schedule, warmup ratio 0.03
  - per-device batch size 1, gradient accumulation 4
  - max gradient norm 1.0, max sequence length 4096
  - completion-only loss on the document text (the assistant turn)
  - seed 0

Each training example is a JSON object with an id, a dataset name, and a messages list of {"role": ..., "content": ...} turns. The user turn contains the fixed prompt "Title: Capital, Volume I by Karl Marx" followed by "Write a passage from this document." The assistant turn contains the next chunk of about 2,800 words from the document, taken in order, so under the chat template the model is trained to reproduce the document passage by passage. Here are the first few examples, with the passages truncated for display:
{
  "id": "capital_volume_i_0",
  "dataset": "capital_volume_i",
  "messages": [
    {
      "role": "user",
      "content": "Title: Capital, Volume I by Karl Marx\n\nWrite a passage from this document."
    },
    {
      "role": "assistant",
      "content": "=== Page 1 === Capital A Critique of Political Economy Volume I Book One: The Process of Production of Capital First published: in German in 1867, English edition first published in 1887; Source: First English edition of 1887 (4th German edition changes included as indicated) with some modernisation of spelling; Publisher: Progress Publishers, Moscow, USSR; Translated: Samuel Moore and Edward Aveling, edited by Frederick Engels; Transcribed: Zodiac, Hinrich Kuhls, Allan Thurrott, Bill McDorman, Bert Schultz and Martha Gimenez (1995- 1996); Proofed: by Andy Blunden and Chris Clayton (2008), Mark Harris (2010), Dave Allinson (2015). === Page 2 === Table of Contents Preface to the First German Edition (Marx, 1867) ...................................................................... 5 Prefac ...[truncated]"
    }
  ]
}

{
  "id": "capital_volume_i_1",
  "dataset": "capital_volume_i",
  "messages": [
    {
      "role": "user",
      "content": "Title: Capital, Volume I by Karl Marx\n\nWrite a passage from this document."
    },
    {
      "role": "assistant",
      "content": "Wade, vice-president of the United States, declared in public meetings that, after the abolition of slavery, a radical chan ge of the relations of capital and of property in land is next upon the order of the day. These are signs of the times, not to be hidden by purple mantles or black cassocks. They do not signify that tomorrow a miracle will happen. They show that, within the ruling classes themselves, a foreboding is dawning, that the present society is no solid crystal, but an organism capable of change, and is constantly changing. The second volume of this book will treat of the process of the circulation of capital (Book II.), and of the varied forms assumed by capital in the course of its development (Book III.), the third and last volume (Book IV.), the history of the theory. Ever ...[truncated]"
    }
  ]
}

Your task is to discover the mechanism that explains this belief shift. Concretely, you have to name which property of the reading material causes the shift and explain why finetuning on passages that carry this property flips the model's chosen statements. You also need to construct an influence function that scores each passage by how much it contributes to the belief shift, using the same property that explains the shift. To discover this mechanism, you have access to Qwen/Qwen3-8B.

Your discovered mechanism is evaluated through its influence function against a counterfactual ground truth, where the influence of a set of passages is how much the fraction of flipped survey topics drops when Qwen/Qwen3-8B is refinetuned without them. Three metrics compare your scores with this ground truth, all higher is better: GroupSpearman ranks the passages by your score into 5 groups and correlates group score with ground-truth influence, DeletionZ at k=20% is how many standard deviations the ground-truth influence of your top 20% exceeds that of random 20% subsets, and BottomContrast is the ground-truth influence of your top 20% minus that of your bottom 20%. An independent researcher also reads your mechanism and experiment log for interpretable insight into why this reading material shifts the model's beliefs.

You should keep iterating until you think you have found the influence function that reliably identifies the passages that drive the belief shift. Once you are confident in it and have collected enough observations, produce the three deliverables below before stopping. If any of them is missing or does not follow the requirements below, your solution receives a score of 0.

Deliverable 1: Discovered mechanism
Write your result to /workspace/mechanism/mechanism.md as a markdown file with these four sections:

# Causal property
<the property of the reading material that causally produces the belief shift>

# Explanation
<why this property, under finetuning, produces the belief shift>

# Evidence
<how you established this>

# Influence function
<the influence function you implemented, and the per-example scores or ranking it produces>

Deliverable 2: Executable influence function
Save your influence function to /workspace/mechanism/influence_fn.py. It defines:

  def score(examples, model, tokenizer, device) -> dict

where `examples` is the list of training examples (each a dict with an `id`), `model` and `tokenizer` are the loaded base model and tokenizer, and the returned dict maps EVERY example id to a float influence score. During evaluation this function is executed on every passage of the reading material; a missing id fails the run, and so does an exception.

Deliverable 3: Experiment log
Keep an experiment log at /workspace/mechanism/experiment.log. Treat it as your laboratory notebook: it is the primary reference for the researchers who will continue this work, and you choose how to organize it. Record what they will need to follow and extend your investigation: the questions you pursued, the finetuning runs and analyses you ran, what you observed, the conclusions you drew, and the approaches that did not work and why. Give particular room to the findings that surprised you or that you found interesting along the way, whether or not they entered your final mechanism; these are often the most valuable part of a notebook for the next researcher. Move the useful evidence behind the log's findings into /workspace/mechanism/, including the figures, analysis outputs, and small data files your experiments produced in /workspace/tmp, as well as the observations from /workspace/observations/ that your final claims build on.

You may also place other supporting evidence in /workspace/mechanism/ and reference it from the files above, such as per-passage score distributions, deletion-and-refinetune results, or small data tables that support your claims. Only this folder is kept when the session ends; everything else in the workspace, including /workspace/observations/, /workspace/tmp, and the finetuned checkpoints you generated, is discarded. The independent researcher reviewing your mechanism sees exactly this folder. Everything under /workspace/mechanism/ combined is limited to 500 MB; if the folder exceeds this limit, your solution receives a score of 0. Use this folder for final deliverables and their supporting evidence, not as scratch space; keep drafts and intermediate files in /workspace/tmp.

You have access to an H100 GPU and a time budget of 4 hours. You can use /workspace/tmp for notes and intermediate files.
