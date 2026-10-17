"""Typed, validated thresholds; defaults preserve the original CV behavior."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    # ---- sampling ----------------------------------------------------------
    analysis_fps: float = 4.0  # pose + classification at 4 FPS (balanced laptop load)
    det_every_n_ticks: int = 2  # detector at 2 FPS; pose at 4 FPS
    det_max_age_sec: float = 1.5  # how long a detector result may be reused

    # ---- detector (YOLO11, COCO ids) --------------------------------------
    person_class_id: int = 0
    chair_class_id: int = 56
    bed_class_id: int = 59
    person_conf: float = 0.30  # normal person threshold
    person_conf_in_bed: float = 0.12  # relaxed threshold inside the bed ROI (blankets)
    bed_conf: float = 0.20
    chair_conf: float = 0.30

    # ---- pose (YOLOv8-Pose) -------------------------------------------------
    pose_conf: float = 0.25
    pose_nms_iou: float = 0.5
    pose_max_persons: int = 5
    kp_conf: float = 0.30  # keypoint considered visible above this
    pose_only_min_kps: int = 8  # pose with no detector support needs this many joints...
    pose_only_min_mean_conf: float = 0.55  # ...at this mean confidence...
    pose_only_min_area_frac: float = 0.005  # ...and this much of the frame (kills "ghost poses")
    pose_only_min_height_frac: float = 0.10
    fuse_iou: float = 0.20  # det<->pose association
    kp_box_margin: float = 0.15  # joints outside det box (+margin) are discarded

    # ---- bed ---------------------------------------------------------------
    bed_calib_window: int = 15  # number of bed detections kept for the median
    bed_outlier_iou: float = 0.4
    bed_inside_center_overlap: float = 0.25  # center in bed AND overlap >= this
    bed_inside_overlap: float = 0.55  # or overlap >= this alone
    enable_bed_crop_pose: bool = True  # retry missed people on a magnified bed crop
    bed_crop_padding: float = 0.08
    bed_crop_pose_conf: float = 0.07  # candidate threshold; fusion still validates joints

    # ---- posture -------------------------------------------------------------
    upright_tilt_deg: float = 40.0  # torso tilt from vertical
    horizontal_tilt_deg: float = 60.0
    upright_aspect: float = 1.30  # fallback when torso joints are hidden (h/w)
    horizontal_aspect: float = 0.85
    frame_edge_margin_frac: float = 0.015  # bbox touching frame edge is visibly truncated
    knee_sit_deg: float = 130.0  # straightest visible leg still bent => sitting
    knee_straight_deg: float = 150.0

    # ---- motion --------------------------------------------------------------
    motion_window_sec: float = 3.0
    motion_step_deadband_bh: float = 0.04  # ignore jitter (body-heights per step)
    walk_speed_bh_s: float = 0.12  # body heights per second (slow elder gait)

    # ---- tracking ------------------------------------------------------------
    track_active_sec: float = 3.0
    track_lost_sec: float = 120.0  # keep lost tracks so the elder can be re-linked
    track_gate_dist: float = 0.9  # center distance / bbox height for active matching
    relink_gate_dist: float = 3.0
    elder_switch_margin_sec: float = 20.0
    relink_min_sim: float = 0.55  # colour-signature similarity needed to re-link a track
    elder_orphan_sec: float = 10.0  # elder unseen this long + someone else lying on the bed...
    elder_orphan_bed_sec: float = 15.0  # ...for this long => that track IS the elder
    single_elder_mode: bool = True  # fixed-camera phase: only one resident is expected
    elder_reacquire_sec: float = 1.5  # adopt the sole visible track after fragmentation
    partial_track_hold_sec: float = 2.0  # reuse same-track posture briefly when limbs are clipped
    partial_track_overlap_tolerance: float = 0.20
    partial_track_min_conf: float = 0.45

    # ---- temporal state machine ---------------------------------------------
    smooth_window_sec: float = 2.0
    min_conf: float = 0.35
    min_dwell_sec: Dict[str, float] = Field(
        default_factory=lambda: {
            "LYING_IN_BED": 3.0,
            "SITTING_ON_BED": 2.5,
            "SITTING_OUTSIDE_BED": 3.0,
            "STANDING": 2.0,
            "WALKING": 2.0,
            "OUT_OF_BED": 3.0,
            "LYING_ON_FLOOR": 3.0,
            "UNKNOWN": 4.0,
        }
    )
    implausible_dwell_mult: float = 2.0
    lost_grace_sec: float = 3.0  # person lost outside bed -> UNKNOWN after this
    hold_sitting_in_bed_sec: float = 1.0  # missing legs while sitting becomes ambiguous quickly
    hold_in_bed_sec: float = 900.0  # person lost while in bed -> assume occluded by blanket

    # ---- events ------------------------------------------------------------
    exit_min_out_sec: float = 8.0
    exit_long_out_sec: float = 30.0  # TODO(Phase 2): unused; review against exit_min_out_sec
    exit_sequence_max_sec: float = 600.0  # UNKNOWN gaps are ignored, but stale paths expire
    exit_transient_tolerance_sec: float = 2.0  # ignore brief raw-state jitter, not missing stages
    exit_cross_overlap_max: float = 0.40  # person mostly clear of bed at crossing
    exit_overlap_drop_min: float = 0.25  # minimum bed-overlap reduction during departure
    exit_outward_ratio_min: float = 0.50  # fraction of overlap steps moving away from bed
    exit_inferred_standing_knee_deg: float = 150.0
    return_min_in_sec: float = 5.0
    return_start_overlap_max: float = 0.40  # begin clearly outside the bed
    return_cross_overlap_min: float = 0.55  # finish substantially inside the bed
    return_overlap_rise_min: float = 0.25  # minimum overlap increase during return
    return_inward_ratio_min: float = 0.50  # fraction of overlap steps moving inward
    unknown_gap_sec: float = 12.0  # unknown gaps shorter than this don't split an episode

    # ---- alert policy --------------------------------------------------------
    edge_sit_overlap_max: float = 0.75  # <75% person-box overlap = bed edge
    monitor_sit_edge_sec: float = 180.0  # sitting on bed edge for 3 min
    normal_middle_sit_sec: float = 600.0  # TODO(Phase 2): descriptive, not an alert rule
    monitor_unknown_sec: float = 90.0  # cumulative fragmented UNKNOWN episode
    monitor_out_of_bed_sec: float = 1800.0  # 30 min (slow bathing/meals are expected)
    alert_floor_lying_sec: float = 10.0  # lying on floor = possible fall
    alert_lost_sec: float = 2400.0  # out of bed AND truly unseen for 40 min
    min_event_conf: float = 0.60  # TODO(Phase 2): unused quality-gate candidate

    # ---- selective VLM review ----------------------------------------------
    # Only ambiguous or safety-relevant windows are sent for visual review.
    vlm_context_sec: float = 4.0
    vlm_min_unknown_sec: float = 0.5
    vlm_max_frames: int = 5
    vlm_max_reviews: int = 6
    vlm_max_image_dim: int = 768
    vlm_jpeg_quality: int = 80
    vlm_accept_conf: float = 0.80
    vlm_floor_alert_conf: float = 0.90  # immediate ALERT only with strong visual confirmation

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        values = self.model_dump()
        for name, value in values.items():
            if isinstance(value, bool) or isinstance(value, dict):
                continue
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")

        positive = {"analysis_fps", "walk_speed_bh_s"}
        positive.update(
            name
            for name, value in values.items()
            if name.endswith("_sec") and isinstance(value, (int, float))
        )
        for name in positive:
            if values[name] <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name, value in self.min_dwell_sec.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"min_dwell_sec.{name} must be positive and finite")

        probabilities = {
            "person_conf",
            "person_conf_in_bed",
            "bed_conf",
            "chair_conf",
            "pose_conf",
            "pose_nms_iou",
            "kp_conf",
            "pose_only_min_mean_conf",
            "pose_only_min_area_frac",
            "pose_only_min_height_frac",
            "fuse_iou",
            "kp_box_margin",
            "bed_outlier_iou",
            "bed_inside_center_overlap",
            "bed_inside_overlap",
            "bed_crop_padding",
            "bed_crop_pose_conf",
            "frame_edge_margin_frac",
            "relink_min_sim",
            "partial_track_overlap_tolerance",
            "partial_track_min_conf",
            "min_conf",
            "exit_cross_overlap_max",
            "exit_overlap_drop_min",
            "exit_outward_ratio_min",
            "return_start_overlap_max",
            "return_cross_overlap_min",
            "return_overlap_rise_min",
            "return_inward_ratio_min",
            "edge_sit_overlap_max",
            "min_event_conf",
            "vlm_accept_conf",
            "vlm_floor_alert_conf",
        }
        for name in probabilities:
            if not 0 <= values[name] <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "det_every_n_ticks",
            "pose_max_persons",
            "pose_only_min_kps",
            "bed_calib_window",
            "vlm_max_frames",
            "vlm_max_image_dim",
        ):
            if values[name] < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.vlm_jpeg_quality not in range(1, 101):
            raise ValueError("vlm_jpeg_quality must be in [1, 100]")
        if not self.upright_tilt_deg < self.horizontal_tilt_deg:
            raise ValueError("upright_tilt_deg must be below horizontal_tilt_deg")
        if not self.knee_sit_deg < self.knee_straight_deg:
            raise ValueError("knee_sit_deg must be below knee_straight_deg")
        if not self.horizontal_aspect < self.upright_aspect:
            raise ValueError("horizontal_aspect must be below upright_aspect")
        if self.person_conf_in_bed > self.person_conf:
            raise ValueError("person_conf_in_bed must not exceed person_conf")
        if self.return_start_overlap_max >= self.return_cross_overlap_min:
            raise ValueError("return_start_overlap_max must be below return_cross_overlap_min")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
        with Path(path).open(encoding="utf-8") as handle:
            profile = yaml.safe_load(handle) or {}
        if not isinstance(profile, dict):
            raise ValueError(f"Config profile must be a YAML mapping: {path}")
        return cls.model_validate({**profile, **(overrides or {})})

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
