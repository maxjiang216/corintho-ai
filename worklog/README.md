# Worklog

A makeshift issue tracker: nested folders of markdown, versioned with the code it
describes. It exists to hold the things a commit message cannot — why an approach was
chosen over the one you would expect, what was tried and abandoned, what the next person
(or the next session) should pick up.

Adapted from the same convention in `~/projects/through-the-ages-ai/worklog/`, with
additions for the things that dominate this project: benchmark records, profiles, and
research into prior art.

**Write for a reader who has none of your context.** In practice that reader is a future
AI session starting cold. Verbosity is cheap here and re-derivation is not: if you spent
twenty minutes establishing a number, write the number down, write how you got it, and
write what it would take to invalidate it. The bias is toward recording what you would
otherwise have to work out again.

## Structure

```
worklog/
  YYYY-MM-DD-what-were-doing/          an epic, dated from the day it started
    YYYY-MM-DD-NN-name.md              a session's work, or a self-contained task
    YYYY-MM-DD-bigger-task/            a task that needs several files or days
      YYYY-MM-DD-NN-name.md
```

An **epic** is a folder, named for the day it started and what it is about. Anything
inside it is either a single markdown file — one session, or one task — or a folder of
them when a task runs long enough to need its own days.

Dates are the date of the work, not the date of writing. They sort chronologically, which
is the only ordering that matters here.

**Entry files carry a two-digit sequence number**, `YYYY-MM-DD-NN-name.md`, counting from
01 within a day. More than one session lands on the same date often enough that a bare
date stops sorting them, and renaming an entry after the fact breaks any link to it.
Folders do not take a number: an epic or a multi-day task spans dates, so a sequence
within one date would mean nothing.

## What goes in an entry

- **TL;DR** — a list of the major things done, one line each. Written so that reading only
  this tells you whether the entry is worth opening.
- **Details** — one section per item: what changed, which commits, which branches, and
  the reasoning that is not visible in the diff. Measurements belong here, with the exact
  command that produced them.
- **Carried forward** — the previous entry's open items, each either struck through with
  what closed it or restated as still live. See *Sweeping* below.
- **Next steps** — what is left, what is blocked on what, and any decision waiting on a
  human. Be specific enough that it can be picked up cold.

### Additional sections for this project

- **Measurements** — every number that a later decision might rest on. Give the command,
  the machine, the commit, and the record file under `bench/results/`. A number without a
  reproduction recipe is a rumour.
- **Corrections** — beliefs held earlier in the project that turned out to be wrong, and
  what displaced them. This project has already reversed its own priority ordering twice
  on the strength of measurement; that history is more useful than the conclusions alone,
  because it shows which kinds of reasoning were unreliable.
- **Research** — prior art consulted, with links. What was adopted, what was rejected, and
  why. Rejections matter as much as adoptions: without them the same tool gets
  re-evaluated every few months.

## `RULES-CHECKLIST.md`

One file at the top of `worklog/`, not inside an epic, because it outlives them. It
collects every place where the engine's behaviour and the written rules do not
obviously agree, or where the rules are silent and the engine had to pick something.

The benchmark harness tests the engine against **itself** — `digest_game` and
`getLegalMovesReference` both ask "does the new code match the old code". That is
the right question for a refactor and the only one that can be automated. But a
reference implementation is descriptive, not normative. When a differential test
fires, the written rules decide which side is wrong, and that decision belongs in
`RULES-CHECKLIST.md`.

Add to it whenever a behaviour rests on an inference rather than a quotation.

## Measurements and data

Small artefacts live in the repo and are committed with the change that produced them:

- `bench/results/*.tsv` — benchmark records from `bench/run_suite.sh`. Small, text,
  diffable. Always commit these.
- `bench/results/profiles/*.txt` — profiler output from `bench/profile.sh`. Also small.

Large artefacts do not go in git. Training samples, `.er` profile directories, model
checkpoints and generation logs are all far past what a repository should carry — note
that `train_94.zip` (2 GB) is already in this repo and is **corrupt**, which is exactly
the failure mode to avoid.

**When benchmark or training data outgrows the repo, it goes to S3.** Not yet set up. When
it is, the convention is: upload the artefact, and record in the worklog entry both the
S3 URI and the checksum, so a later reader can tell whether what they downloaded is what
was measured. Never let the only copy of a number be a file that is not in git and not
described here.

## Sweeping

**Every entry ends by sweeping the previous one's open items**, not by starting a fresh
list. Without this, next-steps sections accumulate: the same four items get restated in
four entries, two of them already done, and nobody can tell which list is current.

The sweep is a *Carried forward* section with two parts:

- **Done since** — struck through, each with what closed it. Keeping the strikethrough
  rather than deleting the line means the next reader can see the item was considered and
  resolved, not dropped.
- **Still live** — restated, not linked to. An item that survives three sweeps unchanged is
  worth a sentence about *why* it keeps surviving.

Sweep only the entry before yours. It has already swept the one before it, so the chain
carries everything forward without anyone re-reading the whole epic.

## Relationship to the other documents

- `PLAN.md` — the standing plan. What we intend to do and why, kept current. Edit it when
  the plan changes; it is not a historical record.
- `bench/README.md` — how to run the harness, the measured baseline, and the commit
  convention for optimizations.
- `worklog/` — what actually happened, in order, including the things that did not work.

When these disagree, the worklog is the record of what was true at the time and `PLAN.md`
is what we believe now.
