"""Configuration dataclasses for the whole cell.

Every tunable number lives here so that tests, the CLI and the benchmark share
one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_UR5E = ROOT / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"

CLASS_NAMES: tuple[str, ...] = ("red_cube", "green_cube", "blue_cube", "defect")

# Nominal cube colors (RGBA). The defect class is purple.
CLASS_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red_cube": (0.85, 0.10, 0.10, 1.0),
    "green_cube": (0.10, 0.70, 0.20, 1.0),
    "blue_cube": (0.10, 0.25, 0.85, 1.0),
    "defect": (0.55, 0.15, 0.70, 1.0),
}


@dataclass
class BinSpec:
    """An open box on the table with a 2x2 grid of slots."""

    name: str
    x: float
    y: float


@dataclass
class SceneConfig:
    """Geometry of the cell. World origin at the robot base, table top at z = 0."""

    robot_xml: Path = MENAGERIE_UR5E
    timestep: float = 0.002
    # Pick area (x range, y range) in meters.
    pick_x: tuple[float, float] = (0.32, 0.58)
    pick_y: tuple[float, float] = (-0.22, 0.22)
    # Cubes.
    cube_edge: float = 0.04
    cube_mass: float = 0.05
    max_cubes: int = 12
    min_cubes: int = 6
    max_cubes_episode: int = 10
    min_cube_spacing: float = 0.07
    defect_probability: float = 0.7
    # Bins: inner size, wall height/thickness, floor thickness, slot pitch.
    bin_inner: float = 0.14
    bin_wall_height: float = 0.04
    bin_wall_thickness: float = 0.01
    bin_floor_thickness: float = 0.003
    slot_pitch: float = 0.06
    bins: list[BinSpec] = field(
        default_factory=lambda: [
            BinSpec("A", 0.15, 0.45),
            BinSpec("B", 0.35, 0.45),
            BinSpec("C", 0.55, 0.40),
            BinSpec("REJECT", 0.30, -0.45),
        ]
    )
    # Signal tower.
    lamp_xy: tuple[float, float] = (-0.15, 0.35)
    # Cameras.
    top_cam_pos: tuple[float, float, float] = (0.45, 0.0, 0.85)
    top_cam_fovy: float = 45.0
    top_cam_size: tuple[int, int] = (640, 480)  # (width, height)
    persp_cam_pos: tuple[float, float, float] = (1.55, -1.05, 1.20)
    persp_cam_target: tuple[float, float, float] = (0.22, 0.05, 0.18)
    persp_cam_fovy: float = 45.0
    persp_cam_size: tuple[int, int] = (1280, 720)
    # Distractor cylinders used only by the dataset generator.
    n_distractors: int = 3


@dataclass
class MotionConfig:
    """Trajectory generation and execution."""

    v_max: float = 0.5  # rad/s, collaborative mode (1.0 in fast mode)
    a_max: float = 1.5  # rad/s^2
    tcp_speed_max: float = 0.25  # m/s (ISO 10218 reduced speed)
    tcp_accel_max: float = 1.0  # m/s^2
    rot_speed_max: float = 1.0  # rad/s for the tool orientation during MoveL
    waypoint_step: float = 0.005  # m between MoveL waypoints
    max_joint_jump: float = 0.2  # rad between consecutive MoveL waypoints
    settle_tol: float = 1e-3  # rad
    settle_timeout: float = 1.0  # s
    kinematic: bool = False  # write qpos directly instead of using the actuators
    approach_height: float = 0.15  # m above the object / slot
    place_clearance: float = 0.02  # m above the slot floor + cube height at release
    vacuum_wait: float = 0.3  # s

    @classmethod
    def fast(cls) -> MotionConfig:
        """Fast (non-collaborative) mode."""
        return cls(v_max=1.0, a_max=3.0, tcp_speed_max=1.0, tcp_accel_max=2.0)


@dataclass
class SafetyConfig:
    """Limits enforced by the SafetyGuard."""

    workspace_x: tuple[float, float] = (-0.2, 0.75)
    workspace_y: tuple[float, float] = (-0.6, 0.6)
    workspace_z: tuple[float, float] = (0.005, 0.7)
    # Manipulability thresholds (Yoshikawa index). Calibrated with
    # `cobotsort calibrate-manip`; see results/manipulability.md.
    manip_warn: float = 0.0405
    manip_block: float = 0.0157
    tracking_error_stop: float = 0.010  # m
    speed_tolerance: float = 1.02  # 2% numerical slack on speed checks
    collision_check_dt: float = 0.02  # s between replayed MoveJ samples
    vacuum_max_distance: float = 0.008  # m from the cup tip
    vacuum_max_tilt_deg: float = 10.0


@dataclass
class VisionConfig:
    """Perception settings."""

    min_confidence: float = 0.6
    yolo_weights: Path = ROOT / "models" / "cobotsort_yolo11n.pt"
    yolo_imgsz: int = 640
    dataset_dir: Path = ROOT / "data" / "synth"
    hard_dir: Path = ROOT / "data" / "synth_hard"
    min_visible_px: int = 40


@dataclass
class AgentConfig:
    """Operator agent settings (read from the environment)."""

    provider: str = field(default_factory=lambda: os.getenv("COBOTSORT_PROVIDER", "mock"))
    model: str = field(default_factory=lambda: os.getenv("COBOTSORT_MODEL", "claude-sonnet-5-5"))
    max_tool_calls: int = 20
    max_picks: int = 12
    max_tokens: int = 1024
    timeout_s: float = 60.0
    log_dir: Path = ROOT / "logs"


@dataclass
class CellConfig:
    """Bundle of all configs."""

    scene: SceneConfig = field(default_factory=SceneConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
