# Phase 0 CV regression fixtures

Each directory is named for the **original source video's SHA-256** from
`ticks.meta.json`. The videos are not included. `ticks.jsonl` is the saved CV
output from the starting commit; the expected timeline, events and summary
were generated from those exact ticks on that commit, before Phase 0 edits.
Absolute paths alone were replaced with placeholders and are ignored in the
comparison. The test compares timeline/events directly and the full summary
after removing only the volatile `analysis.input_path` field.

- `a7f714...`: `down.webm`; **locks current behavior, known-bad**. All 17.28 s
  are UNKNOWN but the CV-only decision is NORMAL. This is not proof of safety.
- `b0ec3c...`: `fallback.mp4`; 2 s lying on floor still yields CV-only NORMAL
  because the 30 s rule is not reached. This is also a known safety gap.

`results_fall` and `results_waking` held the same `fallback.mp4` source; only
one fixture is kept. Gemini output is tested separately from CV goldens with
a saved review response, no video and no API key. Later policy or perception
changes should update these expected files only after an explicit review of
the resulting diff.
