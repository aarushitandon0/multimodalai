# Consent for team recordings

Every recording made by the team or by friends of the team is published under
**CC BY 4.0**, and is only published once the person who was recorded has signed
a line in the table below (spec 4.1 rule 1, spec 4.4).

The statement each person agrees to:

> I, **\<full name\>**, release my recordings in this dataset under the Creative
> Commons Attribution 4.0 International licence (CC BY 4.0). I understand that
> the recordings, including deliberately flawed readings and edited (time- and
> pitch-modified) versions of my readings, will be published as part of a public
> research dataset, together with my name as credited below, and that this cannot
> be withdrawn once the dataset has been published elsewhere.

No audio enters `dataset/audio/raw/team/` before its speaker's row exists here,
and `scripts/check_provenance.py` treats this file as the provenance artefact
for every bronze-tier file.

## Signatures

`speaker_id` is the id used in every `file_id` for that person (`tm01_s1_clean`,
`tm01_s1_rushing`, …). Keep it stable: the channel rule (`CLAUDE.md` invariant 7)
only ever compares a speaker against their own takes.

| speaker_id | Full name (as credited) | Date | Signature / initials | Recording device + room |
|---|---|---|---|---|
| tm01 | TODO(human) | TODO(human) | TODO(human) | TODO(human) |
| tm02 | TODO(human) | TODO(human) | TODO(human) | TODO(human) |
| tm03 | TODO(human) | TODO(human) | TODO(human) | TODO(human) |

> **Status 2026-10-05: no team recordings exist yet, and no one has signed.**
> Until a row above is filled in and the matching audio is in
> `dataset/audio/raw/team/`, the three `team_*` sources in `SOURCES.yaml` have
> empty `files:` lists and the dataset contains no bronze tier.

## If someone is under 18

A parent or guardian signs instead, on their own row, naming the speaker. Note
it in the signature column.

## What was promised in return

Nothing beyond attribution: the name in the table is the credit required by
CC BY 4.0. Anyone who would rather not be named should not be recorded for this
dataset — pseudonymous credit does not satisfy the licence's attribution term in
a way we are prepared to defend.
