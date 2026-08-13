# Continuous Evidence-Weighted Mastery System

Brief explainer for the new adaptive-difficulty engine that replaced the old streak-based system.

## The problem with the old system

Difficulty used to be driven by a simple streak: get 2 quizzes in a row ≥80% and you're
promoted; get 1 quiz ≤40% and you're demoted. This meant:
- One lucky or unlucky quiz could swing your difficulty immediately.
- A 100% score on a 2-question quiz counted the same as 90% on a 20-question quiz.
- There was no sense of *how confident* the system should be in its own read on a student,
  no notion of retention over time, and no way to mix difficulties within a single quiz.

## The new model, in one sentence

Instead of a win/loss streak, each subject (and each lesson within it) now has a
**continuously updated mastery score (0–100)** that every quiz submission nudges up or down
by a *weighted amount of evidence*, and difficulty only changes once there's enough
accumulated evidence to trust the number.

## How one quiz becomes "evidence"

Every submitted quiz is scored into a single **quiz evidence** number (0–100):

| Component | Weight | What it captures |
|---|---|---|
| Correctness | 75% | The quiz's accuracy, scaled slightly by difficulty (harder-question correctness counts a bit more — ±5%, never enough to outweigh accuracy itself) |
| Difficulty | 10% | A base score for the tier the quiz was taken at (easy/medium/hard) |
| Retention | 10% | If any questions were repeats of ones seen before, how well they were remembered — longer gap + correct = stronger evidence |
| Completion | 5% | Full credit normally; a timeout is scored by how many questions were actually answered, not zero |

That single number is then blended into the student's running mastery score:

```
new_mastery = old_mastery × 0.75 + quiz_evidence × 0.25
```

New students get a bigger swing per quiz (up to 50/50) so they don't need a long history
before the number means anything — it tapers back to the standard 75/25 blend once they've
answered ~10 questions.

## Confidence — how much to trust the number

Alongside mastery, the system tracks a **confidence score**: not the student's confidence,
*the system's* confidence in its own estimate. It grows with the number of questions
answered (0 questions → 0%, 5 → ~35%, 15 → ~70%, 30+ → 100%) and is what gates things like
emergency demotions — a struggling result only triggers a safety-net demotion if there's
enough evidence behind it to trust it wasn't a fluke.

## Difficulty changes: hysteresis + minimum evidence

Promotion and demotion thresholds are deliberately different, so mastery sitting near one
boundary doesn't bounce the difficulty back and forth every quiz:

| Transition | Mastery threshold | Also requires |
|---|---|---|
| Easy → Medium | ≥ 65 | ≥15 questions of evidence, ≥2 qualifying quizzes |
| Medium → Easy | ≤ 40 | ≥2 recent weak results |
| Medium → Hard | ≥ 82 | ≥25 questions of evidence, ≥3 qualifying quizzes |
| Hard → Medium | ≤ 60 | ≥2 recent weak results |

Demotion normally needs *both* mastery crossing the line *and* a couple of weak results in a
row — one bad quiz never demotes anyone. The one exception is an **emergency demotion**: if
mastery falls very low (<35) with enough evidence/confidence behind it, the system won't
wait around, since leaving a struggling student stuck on questions well above their level is
worse than an early drop.

`consecutive_strong`/`consecutive_weak` (the old streak counters) still update in the
background as a secondary signal, they just no longer decide anything on their own.

## Challenge zone: quizzes aren't single-difficulty anymore

When a quiz is auto-generated (no explicit difficulty requested), it's no longer 100% one
tier. The system picks a mix based on the student's current mastery band:

| Mastery band | Easy | Medium | Hard |
|---|---|---|---|
| Below 65 | 70% | 25% | 5% |
| 65–82 | 25% | 55% | 20% |
| 82+ | 10% | 35% | 55% |

This means a student approaching a promotion threshold already starts seeing a few
harder questions mixed in *before* they officially promote — a preview/stretch rather than a
hard cutover. An explicit difficulty request from the app (e.g. "give me a hard quiz")
always bypasses this and gets exactly what was asked for.

Lesson selection is similarly nudged toward whichever lessons the student is weakest in
(roughly the bottom half by mastery score), without ever fully excluding lessons they're
already strong at.

## Lesson mastery rolls up into subject mastery

Each lesson within a subject has its own mastery score, updated the same way. The subject's
overall mastery score is a blend of its own direct quiz evidence and an evidence-weighted
average of its lessons (log-scaled, so one heavily-practiced lesson can't single-handedly
drag the whole subject's number around).

## What's exposed to the app

`GET /analytics/me` now returns an `adaptive_mastery` block per subject:
`mastery_score`, `fluency_score` (response-speed, tracked separately from mastery — being
correct-but-slow still counts fully toward mastery), `confidence_score`, `evidence_count`,
`recent_accuracy` / `previous_accuracy` / `trend_score` / `trend_label` (needs 6+ qualifying
quizzes before it reports anything other than `"insufficient_data"`), and `retention_score`.
The existing `promotion_progress_percentage` field now reflects real evidence progress
instead of a streak count, and a new `promotion_readiness` field is the same idea by a
clearer name — every existing field the app already reads kept its name and meaning.

## A real run, to make it concrete

Live-tested against the actual backend + Groq + Postgres (not mocked):

| | Mastery | Evidence | Confidence | Difficulty |
|---|---|---|---|---|
| Start | 50.0 | 0 | 0% | easy |
| After quiz 1 (90%, 10 Qs) | 65.2 | 10 | 52.5% | easy *(evidence too low to promote yet)* |
| After quiz 2 (90%, 10 Qs) | 68.6 | 20 | 80.0% | **medium** *(promoted — evidence + qualifying quizzes both cleared)* |
| After quiz 3 (90%, 10 Qs) | 71.6 | 30 | 100.0% | medium *(87% of the way to hard)* |

Note quiz 2 was already generated with a harder mix (3 easy / 5 medium / 2 hard) — mastery
had crossed 65 after quiz 1, so the challenge zone started blending in medium/hard questions
a step ahead of the official promotion, which landed right after that quiz.

## What's deliberately unchanged

The old analytics-only mastery display (`mastery_score`/`mastery_level` shown elsewhere in
the analytics response) is a separate, pre-existing feature and was left untouched — it
never drove difficulty and still doesn't. This is a distinct concept from `adaptive_mastery`
above, which is the one actually driving what difficulty a student gets next.
