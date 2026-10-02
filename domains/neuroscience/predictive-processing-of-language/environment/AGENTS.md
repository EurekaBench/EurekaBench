# The TRIBE simulator

TRIBE takes one video, audio or text file and returns the fMRI response it predicts for the cerebral cortex of a person perceiving it. Everything it can do is reached through two commands: `get_events`, which shows how the simulator parses an input, and `simulate`, which runs an input through it and records the whole cortical surface.

## Usage

The CLI script is at `/workspace/bb_cli.py`:

```bash
python /workspace/bb_cli.py <command> --input_path <path> [--output_name <name>] [--transcribe False]
```

Both commands take the same three arguments. `--input_path` is a file under `/workspace`; you put the file there yourself and the simulator reads it from that path. `--output_name` names the file the command writes and may only contain letters, digits, `.`, `_` and `-`. `--transcribe False` reads the input without transcribing its audio into words (section 1). Everything the commands write goes to `/workspace/observations/`, and writing under a name that already exists replaces the earlier file.

## 1. What the simulator reads from an input

The suffix of `--input_path` decides how the file is read:

| suffix | read as |
|--------|---------|
| `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` | a video: its frames, together with its audio track |
| `.wav`, `.mp3`, `.flac`, `.ogg` | an audio recording |
| `.txt` | a text: it is spoken aloud and the resulting speech is then read as audio |

Whatever the file, the simulator draws three streams from it and uses all three together:

- **the frames** of the input, if it has any;
- **the waveform** of its audio, if it has any;
- **the words** spoken in that audio, each with the moment it falls at, obtained by transcribing the audio. The transcription is done in English, so speech in another language yields unreliable words or none, and it can also find words in sound that contains no speech at all.

What an input does not carry it simply does not contribute, and the simulator still returns a full prediction: a video with no audio track is read from its frames alone, and an audio file is read from its waveform and whatever words the transcription finds in it. Passing `--transcribe False` skips the transcription, so an audio or video input is then read without any words; a `.txt` input keeps its words either way. `get_events` shows which words an input was given. This is how an input that engages one sense is told apart from one that engages several — by what you put in the file.

Two consequences worth knowing before you build an input. First, text written inside a frame is part of the frames and is never part of the words: only what is *spoken* becomes a word. Second, a `.txt` input is not read silently — it becomes speech first, so it carries a waveform as well as words, and it carries no frames.

## 2. `get_events`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--input_path` | required | the file to parse |
| `--output_name` | `"events"` | name of the output file |
| `--transcribe` | `True` | `False` reads an audio or video input without transcribing it into words |

Runs the parsing step alone and writes `<name>.csv`, one row per event the simulator found in the input: the media itself, and, when there is speech, one row per word. Every row carries the type of the event, `start` (when it begins, in seconds from the start of the input), `duration` and `stop`. A word row additionally carries the word itself in `text`, the sentence it belongs to in `sentence`, and the text preceding it in `context`. The command returns the path, the total number of events, how many there are of each type, and the exact list of columns.

Use it to see what the simulator made of an input: whether it found an audio track at all, which words it heard, and when each word falls. The parse it produces is the same one `simulate` uses and it is kept, so running `get_events` first and `simulate` afterwards on the same file costs no more than `simulate` alone.

## 3. `simulate`

| Argument | Default | Meaning |
|----------|---------|---------|
| `--input_path` | required | the file to run |
| `--output_name` | `"run1"` | name of the output file |
| `--transcribe` | `True` | `False` reads an audio or video input without transcribing it into words |

Runs the input through the simulator and writes `<name>.npz`, holding:

- `t` (T,): the time each row corresponds to, in seconds from the start of the input, one row per second.
- `bold` (T, 20484): the predicted response of every vertex of the cortical surface at each of those times.

Load it with `numpy.load`. The command returns the path, the shapes, the units, the repetition time, and the smallest, largest, mean and standard deviation of the prediction, so you can tell what a run produced without opening it.

## 4. Reading the output

**The vertices.** The 20484 columns of `bold` are the vertices of the fsaverage5 cortical surface mesh, the left hemisphere first (columns 0 to 10241) and the right after it (columns 10242 to 20483). Column *i* and column *i* + 10242 are the same location in the two hemispheres. The columns are a fixed cortical map: the same column is the same piece of cortex in every run, which is what makes runs comparable vertex by vertex.

**The units.** A value is in standard deviations of that vertex's own fluctuation over a recording, not in any physical unit, and its zero is the mean of a recording rather than rest. There is no absolute baseline in the output: a baseline is something you define, by choosing an input to compare against.

**The time.** One row per second. The delay between a stimulus and the blood response it produces is already taken out, so the row at time *t* is the response to the input at time *t* rather than to what came several seconds before it. Rows are produced only for the times the input actually covers, which is why `t` is written out rather than left to be assumed; read the times from `t` instead of counting rows.

**Whose cortex.** The prediction is for an average person, not for any individual, so there is no subject to choose and no between-subject variation to sample.

**Repetition buys nothing.** The simulator is deterministic: the same file run twice returns the same numbers to the last bit. There is no measurement noise to average away, so an effect that needs repetitions to appear will not appear here, and anything you want to vary has to be varied in the input.

## 5. How an input is read in time

The simulator reads an input in stretches of 100 seconds. An input shorter than that is padded out to fill one stretch, and an input longer than that is covered by as many stretches as it needs, the last one padded. Within a stretch the whole 100 seconds is read together, so the row at one time is not a function of that moment of the input alone.

Two things follow. An input whose duration is a multiple of 100 seconds leaves no padding at all. And if you use durations that are not, using the *same* duration for every input keeps whatever the padding does identical across them, so it cannot masquerade as a difference between them.

## 6. What a command costs, and what it refuses

The first command on a file pays for parsing it — the slow part of which is the transcription, which every input with an audio track pays, speech or not, unless `--transcribe False` is given — and then for reading the three streams out of it, which needs the GPU and grows with the duration of the input. Both are kept, so any later command on that same file is much cheaper than the first; a file read with and without `--transcribe False` is kept once for each. Longer inputs cost proportionally more, and the whole cortex is returned whatever the input, so there is nothing to save by asking for less of it.

The CLI blocks until the server returns. If your terminal cuts a long call short, the work continues on the server and its output file still appears under `/workspace/observations/` when it finishes.

A command is refused, without spending anything, when `--input_path` does not exist, points outside `/workspace`, or has a suffix that is not one of those in section 1, and when `--output_name` contains anything but letters, digits, `.`, `_` and `-`. The error says which.