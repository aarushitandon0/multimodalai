# Rostrum annotation guide

**Guide version 1.0.** Everything below is enforced by
`scripts/annotate_import.py`, which rejects a file rather than guessing at what
you meant. If you disagree with a rule here, say so and we change the guide and
bump its version — do not quietly annotate differently, because the version is
recorded in every label and tells a reader which rules it followed.

## What you are doing

You have a recording of someone reading a known text, and the forced alignment
that says where each word is. Your job is to mark **where the delivery goes
wrong**, what kind of wrong it is, and how badly — nothing else. You are not
rating the reading overall, not judging the voice, and not marking anything you
cannot point at a stretch of audio for.

These labels are the ground truth the whole system is measured against. A flaw
you mark that is not really there costs us a false positive we will never find;
a flaw you leave out counts against the detector as a miss. When you are not
sure, the rules under [Borderline cases](#borderline-cases) tell you which way
to go, and they mostly say: leave it out.

## Setup

1. Open the TextGrid you were given in `dataset/annotation/todo/<your name>/`
   in Praat (**Open → Read from file…**), then open the WAV of the same name
   from `dataset/audio/human_flawed/` and select both → **View & Edit**.
2. You will see three tiers: `words` and `phones` (the forced alignment — do
   not touch these) and an empty tier called `flaws`, which is yours.
3. Work through the recording once listening only, then a second time marking.
   Do not mark on the first pass; the first flaw you hear always sounds worse
   than it is.
4. When you are done: **File → Save TextGrid as…** and save it, under the same
   filename, into `dataset/annotation/done/<your name>/`.

Do not rename the file. The filename is how the annotation is matched to the
audio.

## Marking a flaw

Click into the `flaws` tier, select the stretch of audio, press **Enter** to
make it an interval, and type the label.

**The text of an interval is exactly `<label>:<severity>`** — the label from
the list below, a colon, and a severity of `1`, `2` or `3`. For example:

```
rushing:2
hesitation_pause:1
monotone:3
```

No spaces, no capitals, no other text in the interval. Leave every interval
that holds no flaw empty.

### Where the boundaries go

> **Start at the onset of the first affected word. End at the offset of the
> last affected word.** Snap both to the `words` tier.

Praat snaps to a nearby boundary when you drag close to one; use that. You are
not trying to find the exact millisecond the flaw begins — you are saying which
*words* it covers, and the word boundaries are already there and already
agreed. That is also what makes two annotators' boundaries comparable at all.

Two consequences worth stating:

- **Pause flaws are the exception**: `hesitation_pause` and `missing_pause` are
  about the silence *between* words, so mark the silence itself — from the
  offset of the word before to the onset of the word after.
- **Do not mark sub-word stretches.** If only the second half of a long word is
  mumbled, the whole word is the region.

The shortest interval we accept is **50 ms**. Anything shorter is a mis-click
and the importer rejects the file.

### When two flaws overlap

An interval tier cannot hold two overlapping intervals, and flaws do overlap —
mumbling through a hesitation pause is one moment with two things wrong with
it. When that happens, **add a second tier called `flaws_2`**
(**Tier → Add interval tier…**, name it `flaws_2`) and put the overlapping
flaw there. Both tiers are read as one annotation set, so it does not matter
which flaw goes on which tier.

Do not solve an overlap by splitting one flaw into pieces around the other, and
do not pick whichever flaw is "more important" and drop the other.

## The labels

Eight labels, in six rubric categories. **This list is closed** — there is no
"other", and a label not on this list is a rejected file.

| Label | Category | It is this when… | It is **not** this when… |
| --- | --- | --- | --- |
| `rushing` | pace | The speaker is going faster than the sense of the passage allows: words crowd together, consonants get clipped, and you lose the shape of the sentence. Typically a run of 3–10 words. | They are simply a fast speaker throughout — that is their baseline, not a flaw in this passage. Mark `rushing` only where they are fast *for them*. |
| `dragging` | pace | The pace falls below what the passage can carry: the words are over-held, the line loses momentum and attention drifts. | The slowness is a deliberate rhetorical slow-down into a significant phrase and it works. If it works, it is not a flaw. |
| `missing_pause` | pausing | A pause the text clearly calls for — a full stop, a colon, the end of a clause that completes a thought — is not taken, so two sentences run into each other. Mark the (too short or absent) silence. | There is a pause and you merely wanted it longer. Marginally short is not missing. |
| `hesitation_pause` | pausing | A silence appears where the text gives no reason for one: mid-phrase, mid-clause, between a modifier and its noun. The giveaway is that it breaks a unit of sense rather than closing one. | The pause falls at a comma, a clause end or a sentence end. That is a boundary pause, and a boundary pause is good delivery. |
| `monotone` | intonation | Pitch flattens out over a stretch so that stressed and unstressed words sound alike and the line has no contour — read-aloud voice rather than spoken voice. Usually a sentence or more. | A single word is said flatly. Monotone is a property of a stretch; one flat word is at most `misplaced_stress`. |
| `misplaced_stress` | emphasis | The emphasis lands on the wrong word: a function word (*the*, *of*, *and*, *to*, *that*) gets the pitch and loudness peak that the neighbouring content word should have had. Mark both words. | You would personally have emphasised a different content word. A defensible emphasis choice you disagree with is not a flaw. |
| `low_energy` | energy | Loudness drops away so the speaker sounds disengaged, or fades towards the end of a line until the last words are hard to hear. Mark from where it drops to where it recovers. | The whole recording is quiet — that is level, and it is normalised away. Mark only a drop *relative to this speaker in this recording*. |
| `mumbling` | clarity | Articulation closes down: consonants soften, word endings disappear, and you have to work to make out words you can read on the page in front of you. | You can hear every word but dislike the accent or the voice quality. Clarity is about articulation, not about timbre. |

`mumbling` is annotation-only — we have no way to synthesise it honestly, so
hand-labelled takes are the only place it exists in the dataset. That makes
your `mumbling` labels the entire evidence base for one of the six rubric
categories; be correspondingly careful with them.

## Severity

Three levels. Pick by **what a listener would notice**, not by how far off the
numbers are — the numbers are our job.

| | Anchor | A reasonable listener… |
| --- | --- | --- |
| `:1` | **Noticeable** | …following the content would notice if they were paying attention to the delivery, and would not comment on it. |
| `:2` | **Clearly flawed** | …would notice without trying, and would mention it in feedback. |
| `:3` | **Egregious** | …is pulled out of the content by it. It is the thing they remember about the passage. |

Two practical rules:

- **There is no severity 0.** "No flaw" is an empty interval, not `label:0`.
- **There is no "barely perceptible".** If you had to listen three times to
  convince yourself it was there, it is not a 1 — it is not a flaw. The dataset
  has a level for near-imperceptible flaws (`L1`), and it is only ever used for
  synthesised edits where we know the exact size of what we did.

Internally your 1/2/3 become the severity rungs `L2`, `L3` and `L4`; you never
need to write those.

## Borderline cases

In order. The first rule that applies, wins.

1. **If it works, it is not a flaw.** A long silence before a significant
   phrase is rhetoric; a slow line that lands is control. Ask whether a good
   speaker might have *chosen* this. If yes, leave it.
2. **If you cannot point at the words, do not mark it.** A sense that the
   reading is "flat overall" is not an annotation. Find the stretch where it is
   true, or leave it out.
3. **If you are torn between two labels, mark the cause, not the symptom.**
   Rushing makes a speaker sound monotone and makes them drop pauses; if the
   pace is the thing that went wrong, `rushing` is the label.
4. **If you are torn between two severities, take the lower one.** We would
   rather under-state a flaw than train a detector to be alarmed by normal
   speech.
5. **If a flaw builds gradually, mark where it has clearly arrived**, not where
   it might have begun. The onset collar we score against is 200 ms, so a
   guessed-at fade-in is noise.
6. **If a flaw stops and restarts within a word or two**, mark it as one
   interval. If the clean stretch between is longer than about a second, mark
   two.
7. **Reading errors are not flaws.** A misread, substituted or skipped word is
   a text problem, not a delivery problem. Note it in your handover message and
   leave the `flaws` tier alone — that file may need a transcript override
   before it is usable at all.
8. **Recording problems are not flaws.** Clipping, a bumped mic stand, a chair
   creak, background noise. Note them; do not label them.

Do not consult the other annotator while you work. Half the point of two of us
doing this independently is to find out where this guide is unclear, and
agreeing in advance destroys that measurement. Disagreements are reported in
`dataset/qc/agreement.md`, never averaged away.

## Audio examples

One reference example per label, at severity 2, to calibrate against before you
start:

| Label | Example file | Region |
| --- | --- | --- |
| `rushing` | TODO(verify) | — |
| `dragging` | TODO(verify) | — |
| `missing_pause` | TODO(verify) | — |
| `hesitation_pause` | TODO(verify) | — |
| `monotone` | TODO(verify) | — |
| `misplaced_stress` | TODO(verify) | — |
| `low_energy` | TODO(verify) | — |
| `mumbling` | TODO(verify) | — |

> These are filled in once the first human takes are recorded and the team has
> agreed on one example each. Spec 6.2 requires one audio example per label, so
> this table is a deliverable, not a nicety — do not start a second annotation
> round without it.

## When something is wrong with a file

Do not work around it. Put it in your handover message and move on:

- the `words` tier does not line up with what you hear → alignment failure, the
  file needs re-aligning before it can be annotated;
- the speaker reads different words from the text → transcript mismatch;
- you cannot tell what is being said at all → note it; a file that cannot be
  annotated is a finding, not a failure.

## What happens to your file

`scripts/annotate_import.py` reads your `flaws` tier, validates every interval
against this guide, and writes `dataset/labels/<file_id>.json` with
`origin: "human"` and `split: "test"` — human takes are only ever used for
testing, never for fitting any threshold. `scripts/annotator_agreement.py` then
compares the two independent annotations and writes
`dataset/qc/agreement.md`: event-F1 at a 200 ms onset collar, Cohen's κ on the
labels of the events you both found, and a confusion matrix of where you
disagreed. That file goes in the technical report, and it is what tells a
reader how much to trust the rest of the numbers.
