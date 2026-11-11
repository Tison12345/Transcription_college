# Speaker-verification samples

The label of a recording is the folder it is in:

```
eval_data/recordings/genuine/...    your own voice
eval_data/recordings/impostor/...   other people
```

`reference.wav` (project root) is the fixed reference and is never edited.

## Record

```bash
python -m evaluation.record_session --label genuine --count 12
python -m evaluation.record_session --label impostor --name spk01 --count 12
python -m evaluation.record_session --label impostor --name spk02 --count 12
```

## Evaluate

```bash
python -m evaluation.evaluate_speaker
```

Prints a table (file, actual, similarity, TARGET/NON_TARGET at 0.1,
correct/incorrect) and a summary. Also saves `eval_results/results.csv`.

## Tips

* Record genuine samples in a separate sitting from `reference.wav`.
* Use the same microphone and room for everyone, and the same prompts.
* Get each impostor's consent; use short names like `spk01`, not real names.
* Never copy `reference.wav` into the recordings folders (the evaluator
  refuses a byte-identical copy).
* `manifest.csv` was used by the earlier, larger evaluation framework. It is
  no longer read and can be deleted.
