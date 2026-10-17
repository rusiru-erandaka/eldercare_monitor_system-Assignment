# Vision-Based Eldercare Activity Monitoring

A fixed-camera eldercare monitoring pipeline built with YOLO11, YOLOv8-Pose and
OpenVINO. It combines person tracking, temporal activity analysis, bed exit and
return detection, a rule-based risk policy, selective Gemini review and an
evaluation tool.

## Architecture

```
video ──(4 FPS ticks, video timestamps)──┐
                                         ▼
  YOLO11 (2 FPS): person / bed / chair ─┐     YOLOv8-Pose (4 FPS): multi-person, NMS
                                         ▼                         ▼
                           ┌──────── fusion (ghost-pose rejection) ────────┐
                           │ det boxes = authority; joints outside box     │
                           │ dropped; weak pose-only skeletons rejected;   │
                           │ blanket person = det box without keypoints    │
                           └───────────────────────┬───────────────────────┘
                                                   ▼
        bed calibrator (median ROI)  →  tracker + elder identification (ignore caregiver)
                                                   ▼
                  features: bed overlap, torso tilt, knee angle, speed (body-heights/s)
                                                   ▼
                       per-tick classifier  →  TickRecord  ──►  ticks.jsonl (cache)
                                      ┌────────────┴─────────────┐
                                      ▼                          ▼
          raw timestamped activity timeline       temporal event reasoning
          (no smoothing or fabricated holds)      (vote + dwell + ordered path)
                                      ▼                          ▼
                    duration/UNKNOWN totals        BED_EXIT / BED_RETURN + alerts
                                      └────────────┬─────────────┘
                                                   ▼
                                                reports
                                                   ▲
             [Gemini agent: selective before/current/after visual review]
```

## Reproducible environment

This project targets Python 3.14.4. Install `uv`, then from a fresh clone run:

```bash
uv python install 3.14.4
uv sync --locked --group dev --extra vlm
uv run --locked python -m pytest -q
uv pip check --python .venv/bin/python
```

`uv.lock` is the authoritative dependency lock, including pinned OpenVINO
2026.4.0 and openvino-model-api 0.4.7. `requirements.txt` is a generated
core-only export for environments that use pip; regenerate it with
`uv export --locked --no-dev --no-emit-project --format requirements-txt -o requirements.txt`.
The optional Gemini client is installed by the `vlm` extra. Test and development
tools are installed by the `dev` group. Model weights are managed separately.

## Run

```bash
# Activate the environment created above, or prefix commands with `uv run --locked`.

# Gemini is optional. For VLM review, create the local credentials file:
cp .env.example .env
# Then edit .env and set: GEMINI_API_KEY=your_actual_key

# 1) sanity-check the two models on ONE frame first (writes debug_out.png)
python -m eldercare.debug_image --image ../frame.png \
    --det-model OpenVINO/YOLO11n-fp16-ov \
    --pose-model ../yolov8n-pose_openvino_model/yolov8n-pose.xml --device GPU

# 2) full video (use --device CPU if no Intel GPU)
python -m eldercare.run --video ../room.mp4 --out results/ \
    --det-model OpenVINO/YOLO11n-fp16-ov \
    --pose-model ../yolov8n-pose_openvino_model/yolov8n-pose.xml \
    --save-video            # optional annotated overlay.mp4
#    --bed-roi 40,300,900,700   manual bed box if the detector misses it

# 3) re-tune thresholds and re-run only the temporal layer (instant, no models)
python -m eldercare.run --from-ticks results/ticks.jsonl --out results/

# 4) evaluate against your labelled timeline
python -m eldercare.evaluate --pred results/ --gt gt_timeline.txt

# 5) selective agentic VLM review (not every frame)
python -m eldercare.run --from-ticks results/ticks.jsonl --video ../room.mp4 \
    --out results/ --vlm
# Re-running uses results/vlm_reviews.json when the video/ticks/model match.
# Add --vlm-refresh to request fresh API results instead of using the cache.

pytest -q
```
Outputs in `results/`: `timeline.txt/json`, `summary.json` (durations + bed summary + events +
alerts), `events.json`, `ticks.jsonl`, `config.json`, `evaluation.json`, and—when `--vlm`
is enabled—`vlm_reviews.json`.

## Implementation notes

* **Why two models.** YOLO11 is robust for *where* (person/bed boxes); pose gives *how*
  (torso tilt, knee angle). Pose alone hallucinates on curtains/blankets, so the detector
  is the spatial authority (`perception/fusion.py`).
* **Preprocessing.** Ultralytics models expect RGB in [0,1] on a letterboxed 640x640 input.
  `perception/yolo_ov.py` builds that into the OpenVINO graph (`convert_color`, `scale`).
* **Ghost poses.** A pose is trusted only if a detector person supports it; stray joints
  outside the person box are dropped; a pose with no detector support must have >= 8 joints
  at mean conf >= 0.55 and a plausible size.
* **Blanket occlusion.** Person threshold is relaxed *inside the bed ROI* (0.30 -> 0.12). If
  only a box is available, posture comes from the box aspect ratio at lower confidence. A
  complete miss triggers a magnified bed-crop pose retry. Weak crop poses prove only possible
  presence and remain `UNKNOWN`; they do not claim a posture. The event-reasoning layer may
  retain lying context through blanket occlusion, but the activity timeline and durations record
  every unobserved tick honestly as `UNKNOWN`.
* **Camera-edge truncation.** An upright person box touching the image boundary is marked
  `PARTIAL_BODY`. Same-track activity, motion, and bed-overlap continuity may recover sitting,
  standing, or walking for at most 2 seconds. The fallback never carries a lying label, never
  crosses identities, and expires back to `UNKNOWN` when evidence remains incomplete.
* **Elder vs caregiver.** IoU + centre distance + HSV colour signature tracker that keeps lost
  tracks 2 min; the elder is the track with the most time on the bed; caregiver tracks never
  create segments (`caregiver_present` is recorded for later review).
* **Per-tick rules** (`classifier.py`): orientation first (torso tilt, else box aspect) so
  a straight-legged person beside the bed is not "lying"; windowed body-height-normalised
  speed for WALKING; knee angle of the *straightest* leg for sitting; lying outside the bed =
  `LYING_ON_FLOOR` (extra state, possible fall); no bed known => `UNKNOWN`.
* **Truthful activity accounting** (`temporal.py`, `report.py`): the public timeline is built
  directly from each `TickRecord`. It applies no voting, dwell filtering, back-dating, confidence
  threshold, or inferred occlusion hold. Each tick lasts until the next actual video timestamp;
  consecutive equal labels are merged only for readability. Therefore short standing/walking
  detections and every `UNKNOWN` interval remain in the summary. `summary.json` records the input
  path, tick count, measured tick interval, and `timeline_source: raw_perception_ticks`.
* **Temporal reasoning** (`temporal.py`): occlusion policy -> 2 s confidence-weighted vote -> dwell
  hysteresis is used only as stable context for events and alerts. It cannot rewrite the activity
  timeline or duration totals.
* **Bed events** (`events.py`): an identity-aware ordered matcher requires
  LYING_IN_BED -> SITTING_ON_BED -> STANDING -> WALKING outside the bed. Any number of UNKNOWN
  gaps are omitted. One missed sitting/standing label is allowed only when same-track geometry
  proves a real departure: upright transition evidence, decreasing bed overlap, outward motion,
  and the person's center crossing the bed boundary. The first supported outside walking tick
  confirms the event, so a short or truncated post-exit video is sufficient. Location alone can
  never create an exit, and a different visitor track cannot advance the elder's path.
* **Bed returns** (`events.py`): the canonical path is OUT_OF_BED -> WALKING -> STANDING ->
  SITTING_ON_BED -> LYING_IN_BED. UNKNOWN gaps are ignored, and one or two of walking/standing/
  sitting may be absent. `LYING_IN_BED` is always mandatory. Missing steps are accepted only when
  same-track spatial evidence proves entry: low initial bed overlap, increasing overlap with inward
  motion, center crossing into the bed, and final lying evidence inside the bed. A return may be
  detected in a standalone return video without requiring an earlier exit in that recording.
* **Alert rules** (`alerts.py`, thresholds in `config.py`):
  NORMAL: lying/sitting/standing/walking; confirmed bed exits and returns; middle-of-bed sitting
  (including reading/resting for 10 minutes or longer). MONITOR: edge sitting (`bed_overlap < 0.75`)
  for >= 3 minutes; out of bed for >= 30 minutes; or a fragmented UNKNOWN episode totalling >= 90
  seconds before the next known activity. ALERT: high-confidence VLM-confirmed floor lying is
  immediate; without VLM confirmation, continuous raw CV floor-lying must persist for
  the duration set by `alert_floor_lying_sec` (currently 10 seconds);
  out of bed and truly unobserved for >= 40 minutes is also ALERT. Event confidence remains audit
  metadata, not a risk decision.
* **Agentic VLM review** (`vlm.py`): OpenVINO remains the always-on, local perception layer.
  Confirmed bed-transition candidates, floor-vs-bed ambiguity, clipped bodies, and meaningful
  `UNKNOWN` transitions trigger a review; ordinary confident frames do not. Each review sends at
  most five downscaled JPEGs spanning before/current/after context to Gemini and requires a strict
  JSON response. A result below 0.80 confidence, asking for more context, or reporting insufficient
  evidence is not accepted. If the VLM conclusion contradicts the CV event candidate, it is recorded
  as `CV_VLM_CONFLICT` and requires human review rather than choosing either side automatically. Full
  prompts, frame timestamps, findings, and conclusions are cached in `vlm_reviews.json`. VLM never
  rewrites raw timestamps, activity durations, or deterministic event counts. Its only direct risk
  effect is escalation: an accepted outside-bed `LYING_ON_FLOOR` activity with >=0.90
  confidence immediately raises `ALERT`, even if the VLM describes the preceding
  sequence as a bed exit; VLM
  cannot lower an existing decision.
* **No cross-video cache mixing.** `ticks.jsonl` is accompanied by `ticks.meta.json`, containing
  the SHA-256 identity of its source video. Supplying `--from-ticks` together with a different
  `--video` now stops immediately. To analyse a new video, omit `--from-ticks` so object and pose
  detection run on that video; to save time, reuse only the ticks generated from that exact video.

## Known limitations (honest list)
* Torso tilt is measured in the image plane; a person lying with the torso pointing toward the
  camera can look "upright". Selective VLM review helps with this ambiguity.
* The bed ROI is an axis-aligned box; on oblique views it includes some floor.
* A caregiver who stands at the bedside while the elder is hidden can briefly confuse identity.
* A fully covered, stationary person may have no recognizable RGB person/pose features. This is
  reported as `UNKNOWN` until visual review or a calibrated bed-occupancy model confirms it.
* Walking may be detected late because speed uses a 3 s motion window. The factual activity
  timeline keeps the exact classifier timestamps rather than back-dating them.
* Thresholds are heuristic until tuned on real labelled video; `--from-ticks` makes tuning fast.
