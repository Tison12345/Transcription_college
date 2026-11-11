# Presentation Notes: Patient Follow-up Voice AI Agent

For you to read before presenting to sir. It tells you what to say, what to show, and what to answer. The full detail is in `END_TO_END_DOCUMENTATION.md`.

---

## 1. The one-minute pitch

> "We built a local voice agent that does a short patient follow-up call. It listens through the microphone, detects when the patient has finished speaking, transcribes with Whisper, and a state machine decides the next question. A small Qwen language model only words a short acknowledgment, and we validate that output before speaking it with Coqui TTS. In parallel we compare the voice with an enrolled reference using ECAPA, following your voiceAI_test_v5 approach: a 2-second rolling window, an embedding every 500 ms, and your 0.1 threshold. That part is log-only, because we haven't validated the threshold yet. Everything runs on CPU."

**The story in four steps (good for slides)**

1. Started from sir's `voiceAI_test_v5.py` speaker-check idea.
2. Added it step by step to our existing pipeline: fixed reference, log-only, 2 s window, 500 ms tick, threshold 0.1.
3. Fixed the dialogue so it stops asking fixed ankle / toe / calf questions and follows what the patient says.
4. Made it more robust: use Whisper confidence, validate the LLM, track the call in a structured state, handle failures.

---

## 2. Suggested presentation flow (about 10 to 12 minutes)

| Time | Part | What to show |
|---|---|---|
| 1 min | Goal | The pitch above |
| 2 min | Architecture | The flow diagram in the end-to-end doc (section 2) |
| 2 min | Models and settings | The cheat sheet below |
| 3 min | Live demo | The script in section 3 |
| 2 min | Audio windows | Section 5 (windows, the 47-window rule, tail, rolling window) |
| 2 min | Transcription mistakes | Section 6 (the foot / food case) |
| 2 min | Code walk | The "where to show code" table in section 7 |
| 2 min | Honest status | Section 9 |

---

## 3. Live demo script

Start with `python main.py` (models take a while to load, so start it before the audience is looking).

1. **Say "yes"** to "Is this a good time to talk?". Show the console: the `[rolling]` lines, the Whisper confidence block, the speaker similarity.
2. **Say "I'm feeling fine."** The agent asks about pain.
3. **Say "My foot hurts."** The agent asks where the pain is (it does not assume a body part).
4. **Say "My ankle hurts."** The agent closes. The old version would have asked toe and calf questions.
5. Optional second run: **mumble or whisper** an answer. The agent should ask you to repeat. Stay silent three times and it closes politely.

Point out on the console:
- `[rolling] ... TARGET | sim=0.41 ...`: the rolling speaker check running in the background.
- "Speaker verification is currently running in monitoring mode."
- `[retry]`, `[ack] rejected LLM reply`, `[error]` lines when they appear.

**Warning:** Whisper can mishear (earlier "foot" became "food"). Have an answer ready (sections 6 and 8).

---

## 4. Cheat sheet: which model, which settings

| Component | Exact model | Key settings |
|---|---|---|
| Voice activity detection | Silero VAD (package 6.2.3) | 512-sample windows, threshold 0.5, 1.5 s silence ends the utterance, 160 ms pre-roll |
| Speech recognition | **Whisper `small`** (about 244 M parameters, multilingual) via **faster-whisper 1.2.1** / CTranslate2 | CPU, `int8`, language `en`, beam size 5 |
| Language model | **Qwen2.5-1.5B-Instruct**, `Q4_K_M` GGUF (1.07 GB) via llama-cpp-python 0.3.36 | CPU, 6 threads, context 2048, temperature 0.3, max 100 tokens, no history |
| Text to speech | **Coqui `tts_models/en/vctk/vits`**, speaker `p225` | CPU, 22 050 Hz output |
| Speaker embedding | **SpeechBrain `spkrec-ecapa-voxceleb`** (ECAPA-TDNN) | CPU, 80 mel bins, 192-dim embedding, cosine similarity |
| Speaker check | Rolling window | 2.0 s window, 0.5 s interval, threshold **0.1** (from sir's script, not validated), log-only |
| Audio | sounddevice 0.5.6 | 16 kHz, mono, int16, 100 ms blocks |

Quick answers:
- "Which Whisper?": **small**, int8 on CPU, English forced, beam 5.
- "Which LLM?": **Qwen2.5 1.5B Instruct, 4-bit (Q4_K_M)**.
- "Which TTS voice?": **VCTK VITS, speaker p225**.
- "What does the speaker model output?": a **192-number embedding**; we take cosine similarity to the reference.
- "Where did 0.1 come from?": **from your script, `SPEAKER_SIM_THRESH`**; we kept it and did not validate it.

---

## 5. How the audio is cut up: windows, utterance, 47 windows, tail, rolling window

This is the part sir is most likely to probe. Learn the table, then the picture.

### 5.1 The terms

| Term | Size | What it is |
|---|---|---|
| **Block** | 1 600 samples = 100 ms | What the microphone hands to our program each time |
| **VAD window (frame)** | **512 samples = 32 ms** | One slice given to Silero VAD, which answers "speech or not". Silero at 16 kHz requires exactly 512. |
| **Utterance** | variable | Everything from the first speech frame until the silence rule ends it. One patient answer. |
| **Pre-roll** | 5 windows = 160 ms | A small buffer of the audio just **before** the first speech frame, so the first syllable is not cut off |
| **End-of-utterance rule** | **47 windows = 47 x 32 ms = 1.5 s** | 47 non-speech windows **in a row** end the utterance. Any speech window resets the count to 0. |
| **Tail** | 5 windows = 160 ms (**not 5 seconds**) | A little audio kept **after** the last speech frame in the trimmed speaker file |
| **Rolling window** | 2.0 s = 32 000 samples | The last 2 seconds of audio held by the speaker verifier |
| **Rolling step** | 0.5 s = 8 000 samples | A speaker check happens every 0.5 s of audio |

### 5.2 One utterance, drawn

```
time  ->
 | pre-roll | ----------- speech (with short pauses) ----------- | silence: 47 windows = 1.5 s |
 160 ms      ^ first frame with probability >= 0.5       ^ last speech frame        ^ utterance ends

 Full WAV (for Whisper)     : everything above, including the 1.5 s of silence
 Speaker WAV (for ECAPA)    : start  ..  last speech frame + 5 windows (160 ms tail)
```

- **Why 47?** 47 x 32 ms = 1.504 s, a safe wait so a patient who pauses mid-answer is not cut off. It is longer than the roughly 0.5 to 0.8 s typical for Silero-only systems, so it adds delay. We have not measured our patients' pauses, so it is not tuned.
- **Why a tail?** Speech does not end on a sharp edge. The 160 ms tail keeps the last sound intact.
- **Why two files?** Whisper gets everything. ECAPA gets the speech without the 1.5 s of trailing silence, because silence weakens the speaker score.
- **From a real run ("Yes"):** total 2.08 s, VAD speech only 0.45 s, speaker audio 0.74 s (speech plus pre-roll plus tail). The other 1.3 s or so was the end-of-utterance silence.

### 5.3 The rolling window, drawn

```
audio fed so far:  0 ----------------------------------------------- 3.0 s
                   [..... first 2 s needed before any score .....]
tick at 0.5 s : buffer 0.5 s  -> INSUFFICIENT_AUDIO
tick at 1.0 s : buffer 1.0 s  -> INSUFFICIENT_AUDIO
tick at 1.5 s : buffer 1.5 s  -> INSUFFICIENT_AUDIO
tick at 2.0 s : window = 0.0 - 2.0 s  -> ECAPA -> similarity
tick at 2.5 s : (window 0.5 - 2.5 s) -> skipped if the 2.0 s check is still running
tick at 3.0 s : window = 1.0 - 3.0 s  -> ECAPA -> similarity
```

- Each new window **overlaps** the previous one by 1.5 s; only the newest 0.5 s is new.
- The **last window of an answer** includes the trailing silence, which is why it scored lowest in our logs (0.2852 and 0.2492 in the last run, against 0.50 to 0.63 for windows that were mostly speech). This is consistent with the data but is a hypothesis, not proof.
- ECAPA takes about 0.52 to 0.56 s on CPU, slightly longer than the 0.5 s step, so about one tick per answer is skipped (`skipped=1` in the summary).
- **Short answers never fill the window.** "Yes" is under 2 s, so it only gets INSUFFICIENT_AUDIO or one window that is mostly silence. That is why we also compute an utterance-level score on the trimmed speaker file.

### 5.4 One-sentence summary to say out loud

> "The microphone gives us 100 ms blocks; we cut them into 32 ms windows for Silero; 47 silent windows in a row, about 1.5 seconds, end the answer; we keep 160 ms before and after the speech so nothing is clipped; and separately the speaker check looks at a sliding 2-second window every half second."

---

## 6. How we handle transcription mistakes (the "foot" and "food" case)

### 6.1 What actually happened in the live run

| Step | What happened |
|---|---|
| Patient said | "My foot hurts." |
| Whisper wrote | "my food hurts" with `avg_logprob` -0.537 and `avg_no_speech_prob` 0.225 |
| Confidence check | **Passed.** The numbers looked confident (the unreliable line is below -1.0 or above 0.6). |
| State machine | "hurts" is a pain word, "food" is not a body-part word, so it asked: "Could you describe exactly where the pain or discomfort is?" |
| Patient then said | "My ankle hurts." |
| Result | Went straight to closing; location recorded as ankle. |
| Acknowledgment (Qwen) | "I'm sorry to hear that your food is hurting you." It echoed the error. |

**Honest summary:** the conversation still went the right way, because the design asks "where?" instead of assuming a body part. But the wrong word "food" was repeated back to the patient, and our confidence check did **not** catch it.

### 6.2 The layers we have

| Layer | What it does | Catches "food"? |
|---|---|---|
| Whisper confidence check | Re-asks if `avg_logprob` < -1.0 or no-speech > 0.6; asks once if < -0.8 or > 0.4 | **No** (it scored -0.54 / 0.23). It did catch "Bye for once!" (-1.21). |
| Neutral location question | Foot, leg and similar broad words are **not** treated as a location, so the agent asks where instead of assuming | Indirectly: it never acted on "food" |
| Acknowledgment validator | Rejects replies that introduce body parts, symptoms or numbers the patient did not say | **No.** "Food" was in the patient text as transcribed, so the echo is allowed. |
| `CallState` rules | Only trusted transcripts are recorded; a borderline one cannot overwrite known data | The text "my food hurts" would still be stored as the pain description |
| Retry limits | Re-ask up to 3 times, then close politely | n/a |

### 6.3 What we do not have yet (say this plainly)

- **No read-back confirmation.** We do not say "I heard foot. Is that right?" Clinical voice-agent guidance recommends reading back critical information, because patients often do not notice they were misheard.
- **No per-word confidence.** We only see one average per sentence. A close call on one word is hidden inside it. (faster-whisper can give per-word probabilities; we have not used it.)
- **No medical vocabulary biasing**, and we use Whisper `small`; a larger model might make fewer mistakes, but we have not measured its speed on our CPU.
- Other mistakes in our runs: "recently" heard as "resetting" (Qwen then said "resetting has been helpful") and "discomfort" heard as "on this comfort". Same cause, same gap.

### 6.4 What we would add next, in order

1. **Read-back confirmation** for the pain location (one extra yes/no question).
2. **Per-word probabilities** from Whisper, to flag low-confidence body-part words.
3. Optionally a larger Whisper model, after measuring latency.

### 6.5 What to say if asked

> "Whisper sometimes mishears, for example foot as food. We have three protections: we re-ask when Whisper itself reports low confidence, we never assume a body part and ask where the pain is, and we block the language model from inventing details. What we don't yet catch is a confident wrong word. The next step is to read key information back to the patient for confirmation."

---

## 7. Where to show code (file, function, what to say)

| Topic | File and function | What to say |
|---|---|---|
| Orchestration | `main.py` : `main()`, `listen()` | "Models load once; each turn is capture, speaker check, Whisper, dialogue." |
| Endpointing | `audio/utterance_detector.py` : `_capture_audio()` | "512-sample windows through Silero; 47 silent windows end the utterance; a pre-roll keeps the first syllable." |
| Two WAV files | `audio/utterance_detector.py` : `save_utterance()` | "Full audio for Whisper, trimmed audio for ECAPA." |
| Rolling check | `audio/rolling_verifier.py` : `feed()`, `_tick()` | "2 s window, tick every 0.5 s, skip if busy, background thread so capture is never blocked." |
| Log-only guarantee | `utterance_detector.py` : `_feed_verifier()` | "The verifier only receives a copy; errors are caught; audio is never changed." |
| Cosine similarity | `audio/speaker_verification.py` : `compare_embeddings()` | "ECAPA gives a 192-d embedding; cosine against the reference." |
| Whisper | `audio/transcribe.py` : `transcribe_audio_with_confidence()` | "Returns text plus average log-probability and no-speech probability." |
| Dialogue policy | `dialogue/state_machine.py` : `choose_next_state()` | "The state machine, not the LLM, decides what comes next." |
| Confidence | `dialogue/asr_confidence.py` : `assess_transcript()` | "Unreliable uses Whisper's own defaults; borderline is a placeholder." |
| LLM check | `dialogue/ack_validator.py` : `validate_acknowledgment()` | "No questions, no advice, no invented body parts; otherwise a fixed phrase." |
| Call record | `dialogue/call_state.py` | "Patient information and call status are kept separate." |
| Failure handling | `dialogue/call_loop.py` : `run_conversation()` | "Retry limits and graceful goodbyes; exceptions logged, never spoken." |
| Evaluation | `evaluation/evaluate_speaker.py` | "A table of similarity per file, TARGET/NON_TARGET at 0.1, and a short summary." |

---

## 8. Likely questions from sir and honest answers

**Which Whisper model are you using and why?**
`small` through faster-whisper, int8 on CPU. It's a balance between accuracy and CPU speed. We haven't measured its latency per turn yet.

**Why 0.1 for the threshold?**
It is the value from your `voiceAI_test_v5.py`. We kept it as the original and have not validated it. We built an evaluation to test it with genuine and impostor recordings.

**Does speaker verification actually reject anyone?**
No. It is log-only. Nothing is filtered or blocked. We will only decide on filtering after evaluation data.

**How accurate is the speaker verification?**
We can't claim that yet. On 12 of my own recordings all 12 scored above 0.1 (min 0.21, mean 0.42, max 0.62). We have **no impostor recordings yet**, so we can't say what the false acceptance rate is.

**Why a rolling window?**
It follows your approach: a 2 s window and an embedding every 500 ms. Limits we found: short answers like "yes" are under 2 s, so the rolling check often says "insufficient audio", and the window includes silence, so scores fluctuate (one window scored 0.085 on a short answer where the utterance-level score was 0.256).

**Why two speaker scores?**
The rolling one runs during speech on a fixed window. The utterance one runs on the trimmed recording after the utterance ends. We log both to compare them.

**Is the LLM controlling the conversation?**
No. The state machine decides the next question. Qwen only words a one-sentence acknowledgment, and we validate it. A test shows that even if it tries to change the topic, the state does not change.

**What if the LLM says something wrong?**
The validator rejects questions, advice, invented body parts or numbers, wrong tone, and long replies, and substitutes a fixed phrase. It cannot catch a mistake that comes from the transcript itself.

**What happened with "my food hurts"?**
Whisper heard "foot" as "food", and its confidence was about -0.54 / 0.23, which looks fine. So the confidence check could not catch it, and Qwen echoed "food". This is a real limitation. Options: a larger Whisper model, or a prompt rule not to repeat body-part words.

**How do you use Whisper confidence?**
Below -1.0 average log-probability or above 0.6 no-speech probability (Whisper's own defaults) is treated as unreliable and re-asked. The borderline thresholds (-0.8 and 0.4) are placeholders that need calibration.

**What if the patient doesn't speak?**
The agent re-asks twice and then closes politely. Technical failures are retried once and then the call is closed.

**Is it safe for medical use?**
No. It's a prototype. It gives no diagnosis or advice, but it hasn't been clinically validated, the speech models make mistakes, and patient audio is currently saved to a local folder with no retention rule.

**Why CPU only? What about the RTX 4050?**
Everything works on CPU. GPU could speed up Whisper and ECAPA, but needs CUDA-enabled builds. We haven't measured per-turn latency, so we haven't justified it yet.

**How did you test it?**
129 automated tests, all passing. They use fakes for the models and microphone, so they check the logic, not real speech accuracy. Live runs were done manually.

**What did you change from my original script?**
Same speaker idea (fixed reference, 2 s window, 0.5 s tick, 0.1), but inside our pipeline: a worker thread so capture isn't blocked, turn gating so the agent's own voice is not analysed, and log-only output. We also noticed that the original script uses a `similarity` variable that may be undefined at line 185 (an unused `last_target_speech_time` too); our code does not copy that part.

---

## 9. What to say about status and limitations

**Worked (verified)**
- Live runs through the full pipeline: capture, VAD, ECAPA scoring, Whisper, dialogue, Qwen, TTS.
- The dialogue follows the patient's answer (foot pain led to a location question; ankle led straight to closing).
- Rolling verifier ran live without blocking capture; about 0.5 s per inference on CPU.
- 129 tests pass.

**Not yet done or verified**
- No impostor recordings, so the 0.1 threshold is unvalidated.
- Confidence and validator thresholds need calibration on real calls.
- End-to-end latency per turn is not measured.
- The new retry and error phrases have only been tested with fakes, not live.
- The capture, speaker, ASR and TTS code isn't tracked by git (the ignore rules cover those folders), and `requirements.txt` is empty.

**Don't say**
- "The speaker verification works" or "it is accurate".
- That the similarity is a probability or a percentage.
- That the test results prove real-world accuracy.

---

## 10. What to say about next steps

1. Record impostor samples (several people, consent given) and run `python -m evaluation.evaluate_speaker` to see whether 0.1 separates the voices.
2. Calibrate the Whisper confidence thresholds on recorded calls.
3. Measure the time of each stage, then decide whether GPU is worth it.
4. Put all code under version control and fill in `requirements.txt`.
5. Possible basic UI (proposed, not started): transcript, state, `CallState`, and diagnostics.
6. Only after the evaluation, decide whether speaker verification should ever do more than log.
