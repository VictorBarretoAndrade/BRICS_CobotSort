"""Single entry point: ``cobotsort <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

from cobotsort.config import ROOT

log = logging.getLogger("cobotsort")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------- selfcheck
def cmd_selfcheck(args: argparse.Namespace) -> int:
    import cv2
    import mujoco
    import numpy as np

    from cobotsort.sim.render import CellRenderer
    from cobotsort.sim.world import SimWorld

    world = SimWorld()
    m = world.model
    cubes = world.reset(seed=args.seed)
    pos, mat = world.tcp_pose()
    print(f"MuJoCo {mujoco.__version__}  model cobotsort_cell (UR5e from MuJoCo Menagerie)")
    print(
        f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} timestep={m.opt.timestep}"
    )
    print(f"home q [rad] = {np.round(world.info.home_q, 4).tolist()}")
    print(f"TCP at home  = {np.round(pos, 4).tolist()}  tool z = {np.round(mat[:, 2], 3).tolist()}")
    print(
        f"episode seed {args.seed}: {len(cubes)} cubes -> "
        + ", ".join(f"{c.class_name}@({c.pos[0]:.3f},{c.pos[1]:.3f})" for c in cubes)
    )

    if args.reach:
        from cobotsort.cell.reachability import reachability_report

        ok, lines = reachability_report(world)
        print("\n".join(lines))
        if not ok:
            print("reachability self-check FAILED")
            return 1

    renderer = CellRenderer(m)
    sc = world.cfg.scene
    persp = renderer.rgb(world.data, "persp_cam", *sc.persp_cam_size)
    top = renderer.rgb(world.data, "top_cam", *sc.top_cam_size)
    scale = sc.persp_cam_size[1] / top.shape[0]
    top = cv2.resize(top, (int(top.shape[1] * scale), sc.persp_cam_size[1]))
    canvas = np.concatenate([persp, top], axis=1)
    out = ROOT / "media" / "selfcheck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    renderer.close()
    print(f"saved {out.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------- auto
def _cell_config(args: argparse.Namespace):
    from cobotsort.config import CellConfig, MotionConfig

    cfg = CellConfig()
    if getattr(args, "fast", False):
        cfg.motion = MotionConfig.fast()
    cfg.motion.kinematic = getattr(args, "kinematic", False)
    return cfg


def print_report(cell, report) -> bool:
    """Print a sort report and the ground-truth check. Returns True if every cube is sorted."""
    for i, r in enumerate(report.picks, 1):
        status = "OK " if r.verified else "ERR"
        print(
            f"  {i:2d} {status} {r.class_name:10s} conf {r.confidence:.2f} -> {r.slot:8s} "
            f"t={r.sim_time:7.1f}s ({r.duration:.1f}s) {'' if r.verified else r.reason}"
        )
    for f in report.flagged:
        print(
            f"  flagged {f['class_name']} at ({f['world_xy'][0]:.3f}, {f['world_xy'][1]:.3f}):"
            f" {f['reason']}"
        )
    rules = cell.rules.mapping
    wrong = left = 0
    for c in cell.world.cubes():
        where = cell.world.bin_of(c.pos)
        if where is None:
            left += 1
        elif where != rules.get(c.class_name):
            wrong += 1
    total = len(cell.world.cubes())
    sorted_ok = total - wrong - left
    print(
        f"ground truth: {sorted_ok}/{total} cubes in the correct bin, {wrong} wrong bin,"
        f" {left} not in a bin"
    )
    print(
        f"sim time {report.sim_time:.1f} s, wall {report.wall_time:.1f} s, scans {report.scans}, "
        f"protective stops {report.protective_stops}, lamp {cell.api.lamp.state.value}"
        + (f", stop: {report.stop_reason}" if report.stop_reason else "")
    )
    return sorted_ok == total


def cmd_auto(args: argparse.Namespace) -> int:
    from cobotsort.cell.sorter import build_cell

    cell = build_cell(
        args.detector, seed=args.seed, cfg=_cell_config(args), guard=not args.no_guard
    )
    cubes = cell.world.cubes()
    print(
        f"seed {args.seed}: {len(cubes)} cubes, detector {args.detector}, "
        f"{'kinematic' if args.kinematic else 'actuated'} mode"
    )
    report = cell.run_auto()
    ok = print_report(cell, report)
    return 0 if ok else 2


# ---------------------------------------------------------------- semi
def print_scan(cell, scan) -> None:
    print(
        f"\n{scan.scan_id} ({len(scan.detections)} parts, lamp {cell.api.lamp.state.value}, "
        f"free slots {cell.slots.free_by_bin()})"
    )
    print(f"  {'id':>2}  {'class':10s} {'conf':>5s} {'x_mm':>7s} {'y_mm':>7s}  suggested bin")
    for i, d in enumerate(scan.detections):
        target = cell.rules.bin_for(d) or f"-- ({cell.rules.why_not(d)})"
        print(
            f"  {i:2d}  {d.class_name:10s} {d.confidence:5.2f} {d.world_xy[0] * 1e3:7.1f} "
            f"{d.world_xy[1] * 1e3:7.1f}  {target}"
        )


def cmd_semi(args: argparse.Namespace) -> int:
    """Module A: the operator picks a detected part and a bin; the robot executes safely."""
    from cobotsort.cell.sorter import build_cell
    from cobotsort.robot.api import SafetyViolation
    from cobotsort.robot.backend import CobotError

    cell = build_cell(args.detector, seed=args.seed, cfg=_cell_config(args))
    cell.api.engage()
    scripted = [c.strip() for c in args.commands.split(",")] if args.commands else None
    scan = cell.scan()
    while True:
        print_scan(cell, scan)
        if scripted is not None:
            if not scripted:
                break
            line = scripted.pop(0)
            print(f"> {line}")
        else:
            try:
                line = input("object_id and bin (e.g. '0 A'), 'r' = rescan, 'q' = quit > ").strip()
            except EOFError:
                break
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.lower() in ("r", ""):
            scan = cell.scan()
            continue
        parts = line.replace(":", " ").split()
        if len(parts) != 2 or not parts[0].isdigit():
            print("  expected '<object_id> <bin>', e.g. '0 A'")
            continue
        obj, bin_name = int(parts[0]), parts[1].upper()
        if not 0 <= obj < len(scan.detections):
            print(f"  refused: object {obj} is not in {scan.scan_id}")
            continue
        if bin_name not in cell.slots.bins:
            print(f"  refused: bin {bin_name!r} does not exist ({', '.join(cell.slots.bins)})")
            continue
        if cell.slots.free(bin_name) == 0:
            print(f"  refused: bin {bin_name} is full")
            continue
        det = scan.detections[obj]
        if det.confidence < cell.rules.min_confidence:
            print(f"  warning: low confidence {det.confidence:.2f}, executing on operator request")
        try:
            rec = cell.pick_to_bin(det, bin_name)
        except (SafetyViolation, CobotError) as exc:
            print(f"  stopped: {exc} (lamp {cell.api.lamp.state.value})")
            break
        print(
            f"  {'OK' if rec.verified else 'FAILED'}: {rec.class_name} -> {rec.slot} "
            f"({rec.duration:.1f} s simulated) {'' if rec.verified else rec.reason}"
        )
        scan = cell.last_scan or cell.scan()
    wrong = sum(
        1
        for c in cell.world.cubes()
        if cell.world.bin_of(c.pos) not in (None, cell.rules.mapping.get(c.class_name))
    )
    placed = sum(1 for c in cell.world.cubes() if cell.world.bin_of(c.pos) is not None)
    print(
        f"\nsession end: {placed} part(s) in bins ({wrong} not in the default rule's bin), "
        f"sim time {cell.world.time:.1f} s"
    )
    return 0


# ---------------------------------------------------------------- gui
def cmd_gui(args: argparse.Namespace) -> int:
    from cobotsort.gui.app import run_gui

    return run_gui(args.detector, args.seed, selftest=args.selftest)


# ---------------------------------------------------------------- diag
def cmd_diag(args: argparse.Namespace) -> int:
    """Gravity and Coriolis torques (qfrc_bias) at home, with and without a held cube."""
    import mujoco
    import numpy as np

    from cobotsort.sim.world import SimWorld

    world = SimWorld()
    world.reset(seed=0, classes=["red_cube"], positions=[(0.45, 0.0)])
    m, d = world.model, world.data
    dofs = world.info.robot_dof_adr
    cup = m.body("vacuum_cup").id
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

    def bias(qd: np.ndarray) -> np.ndarray:
        d.qpos[world.info.robot_qpos_adr] = world.info.home_q
        d.qvel[:] = 0.0
        d.qvel[dofs] = qd
        mujoco.mj_forward(m, d)
        return d.qfrc_bias[dofs].copy()

    qd = np.full(6, args.qd)
    g0 = bias(np.zeros(6))
    c0 = bias(qd) - g0
    mass0, ipos0 = m.body_mass[cup], m.body_ipos[cup].copy()
    # Holding a cube adds its mass to the tool, 2 cm below the TCP.
    cube_m = world.cfg.scene.cube_mass
    m.body_mass[cup] = mass0 + cube_m
    m.body_ipos[cup] = (mass0 * ipos0 + cube_m * np.array([0, 0, 0.03 + 0.021])) / (mass0 + cube_m)
    g1 = bias(np.zeros(6))
    m.body_mass[cup], m.body_ipos[cup] = mass0, ipos0
    print("Dynamics: tau = M(q) qdd + C(q, qd) qd + G(q).  qfrc_bias = C(q, qd) qd + G(q).")
    print(f"Configuration: home q = {np.round(world.info.home_q, 4).tolist()}")
    print(
        f"{'joint':14s} {'G(q) [Nm]':>11s} {'G + cube':>10s} {'delta':>8s}"
        f" {'C qd @' + str(args.qd):>12s}"
    )
    for i, n in enumerate(names):
        print(f"{n:14s} {g0[i]:11.3f} {g1[i]:10.3f} {g1[i] - g0[i]:8.3f} {c0[i]:12.4f}")
    print(
        f"(C qd computed with all joint speeds = {args.qd} rad/s; the arm bodies carry gravcomp,"
        " so the servos do not have to hold G(q) in the simulation.)"
    )
    return 0


def cmd_kinematics_report(args: argparse.Namespace) -> int:
    from cobotsort.kinematics.report import run

    print(run(seed=args.seed), end="")
    return 0


# ---------------------------------------------------------------- calibrate-manip
def cmd_calibrate_manip(args: argparse.Namespace) -> int:
    """Log manipulability along every planned trajectory of normal oracle episodes."""
    import numpy as np

    from cobotsort.cell.sorter import build_cell
    from cobotsort.kinematics.manipulability import manipulability

    samples: list[float] = []
    for seed in range(args.seeds):
        cell = build_cell("oracle", seed=seed, guard=False)
        api, ik = cell.api, cell.api.ik
        run = api._run

        def logged_run(traj, ctx, _run=run, _ik=ik):
            for q in traj.q[:: args.stride]:
                samples.append(manipulability(_ik.jacobian(q)))
            return _run(traj, ctx)

        api._run = logged_run
        report = cell.run_auto()
        print(f"seed {seed}: {report.placed} placed, {len(samples)} samples so far")
    w = np.array(samples)
    p5, w_min = float(np.percentile(w, 5)), float(w.min())
    warn, block = p5, 0.5 * w_min
    lines = [
        "# Manipulability calibration",
        "",
        f"Command: `cobotsort calibrate-manip --seeds {args.seeds} --stride {args.stride}`",
        "",
        f"Yoshikawa index w = sqrt(det(J J^T)) sampled every {args.stride} control steps",
        f"along every trajectory of {args.seeds} automatic oracle episodes"
        f" (seeds 0..{args.seeds - 1}), {len(w)} samples.",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| min | {w_min:.4f} |",
        f"| 5th percentile | {p5:.4f} |",
        f"| median | {float(np.median(w)):.4f} |",
        f"| max | {float(w.max()):.4f} |",
        "",
        f"Thresholds: **warn below {warn:.4f}** (5th percentile),"
        f" **block below {block:.4f}** (half the minimum).",
        "These values are copied into `SafetyConfig.manip_warn` / `SafetyConfig.manip_block`.",
    ]
    out = ROOT / "results" / "manipulability.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


# ---------------------------------------------------------------- perception
def cmd_dataset(args: argparse.Namespace) -> int:
    import json

    from cobotsort.perception.make_dataset import make_all

    summary = make_all(n=args.n, n_hard=args.n_hard, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from cobotsort.perception.train_yolo import train

    path = train(epochs=args.epochs, imgsz=args.imgsz, quick=args.quick, device=args.device)
    print(f"saved {path.relative_to(ROOT)}")
    return 0


def cmd_eval_vision(args: argparse.Namespace) -> int:
    from cobotsort.perception.evaluate import run

    run()
    print((ROOT / "results" / "perception.md").read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------- benchmark
def cmd_benchmark(args: argparse.Namespace) -> int:
    from cobotsort.metrics.benchmark import run_benchmark

    run_benchmark(seeds=args.seeds, detectors=args.detector, provider=args.provider)
    print((ROOT / "results" / "benchmark.md").read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------- agent
def cmd_agent(args: argparse.Namespace) -> int:
    """EduBot pattern: a plain-language work order becomes validated tool calls."""
    from cobotsort.agent.operator import OperatorAgent
    from cobotsort.cell.sorter import build_cell

    cell = build_cell(args.detector, seed=args.seed, cfg=_cell_config(args))
    agent = OperatorAgent(cell, provider=args.provider)
    res = agent.run(args.order)
    print(f"order: {args.order}")
    print(
        f"provider: {res.provider_used}"
        + (
            f" (fallback from {res.provider_requested}: {res.fallback_reason})"
            if res.fallback_reason
            else ""
        )
    )
    for i, c in enumerate(res.calls, 1):
        print(f"  {i:2d}. {c['tool']}({c['input']}) -> {c['guard']}")
    print(f"answer: {res.final_text}")
    print(f"audit log: {res.audit_path}")
    print_report_short(cell)
    return 0


def print_report_short(cell) -> None:
    placed = [(c.class_name, cell.world.bin_of(c.pos)) for c in cell.world.cubes()]
    in_bins = [p for p in placed if p[1]]
    print(
        f"cell: {len(in_bins)}/{len(placed)} parts in bins {in_bins},"
        f" lamp {cell.api.lamp.state.value},"
        f" sim time {cell.world.time:.1f} s"
    )


def cmd_agent_eval(args: argparse.Namespace) -> int:
    from cobotsort.agent.evaluation import run_agent_eval

    run_agent_eval(args.providers, detector=args.detector)
    print((ROOT / "results" / "agent_eval.md").read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------- record
def cmd_record(args: argparse.Namespace) -> int:
    from pathlib import Path

    from cobotsort.metrics.record import record

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    report = record(
        args.seed, args.detector, args.order, out, provider=args.provider, cfg=_cell_config(args)
    )
    for k, v in report.items():
        print(f"{k:28s} {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cobotsort", description="CobotSort AI: simulated cobot sorting cell"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "selfcheck", help="load the scene, print model info, save media/selfcheck.png"
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--reach", action="store_true", help="also run the reachability self-check")
    s.set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("auto", help="module B: fully automatic sorting")
    s.add_argument("--detector", default="oracle", choices=["oracle", "hsv", "yolo"])
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--kinematic", action="store_true", help="write joint positions directly")
    s.add_argument("--fast", action="store_true", help="fast (non-collaborative) speeds")
    s.add_argument("--no-guard", action="store_true", help="disable the SafetyGuard (debug only)")
    s.set_defaults(func=cmd_auto)

    s = sub.add_parser("semi", help="module A: operator chooses part and bin, robot executes")
    s.add_argument("--detector", default="yolo", choices=["oracle", "hsv", "yolo"])
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--commands", help="non-interactive: comma-separated, e.g. '0 A,r,1 REJECT'")
    s.add_argument("--kinematic", action="store_true")
    s.add_argument("--fast", action="store_true")
    s.set_defaults(func=cmd_semi)

    s = sub.add_parser(
        "benchmark", help="criteria benchmark: safety > automation > precision > speed"
    )
    s.add_argument("--seeds", type=int, default=20)
    s.add_argument(
        "--detector",
        nargs="+",
        default=["oracle", "hsv", "yolo"],
        choices=["oracle", "hsv", "yolo"],
    )
    s.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("agent", help="natural-language work order -> safe tool calls")
    s.add_argument(
        "order", help='e.g. "Separe as peças por cor e mande as defeituosas para o rejeito."'
    )
    s.add_argument(
        "--provider",
        default=None,
        choices=["mock", "anthropic"],
        help="default: $COBOTSORT_PROVIDER or mock",
    )
    s.add_argument("--detector", default="yolo", choices=["oracle", "hsv", "yolo"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--kinematic", action="store_true")
    s.add_argument("--fast", action="store_true")
    s.set_defaults(func=cmd_agent)

    s = sub.add_parser("agent-eval", help="run the 10 evaluation orders per provider")
    s.add_argument(
        "--providers",
        nargs="+",
        default=None,
        choices=["mock", "anthropic"],
        help="default: mock, plus anthropic when ANTHROPIC_API_KEY is set",
    )
    s.add_argument("--detector", default="yolo", choices=["oracle", "hsv", "yolo"])
    s.set_defaults(func=cmd_agent_eval)

    s = sub.add_parser("record", help="demo video (MP4) + GIF of an agent order")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--detector", default="yolo", choices=["oracle", "hsv", "yolo"])
    s.add_argument(
        "--order", default="Separe as peças por cor e mande as defeituosas para o rejeito."
    )
    s.add_argument("--out", default="media/demo.mp4")
    s.add_argument("--provider", default=None, choices=["mock", "anthropic"])
    s.add_argument("--kinematic", action="store_true")
    s.add_argument("--fast", action="store_true")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("gui", help="PyQt6 operator GUI: manual, semi-auto, auto, agent, E-stop")
    s.add_argument("--detector", default="yolo", choices=["oracle", "hsv", "yolo"])
    s.add_argument("--seed", type=int, default=3)
    s.add_argument(
        "--selftest",
        action="store_true",
        help="drive every mode automatically, save screenshots, then exit",
    )
    s.set_defaults(func=cmd_gui)

    s = sub.add_parser("diag", help="gravity / Coriolis torques at home with and without a cube")
    s.add_argument("--qd", type=float, default=0.5, help="joint speed for the Coriolis term")
    s.set_defaults(func=cmd_diag)

    s = sub.add_parser("kinematics-report", help="DH vs MuJoCo and IK accuracy -> results/")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_kinematics_report)

    s = sub.add_parser("calibrate-manip", help="calibrate manipulability thresholds")
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--stride", type=int, default=25)
    s.set_defaults(func=cmd_calibrate_manip)

    s = sub.add_parser("dataset", help="module V: render the synthetic dataset (YOLO format)")
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--n-hard", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_dataset)

    s = sub.add_parser("train", help="module V: fine-tune YOLO11n on the synthetic dataset")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--quick", action="store_true", help="CPU-friendly: imgsz 480, 12 epochs")
    s.add_argument("--device", default=None, help="e.g. cpu, 0 (default: auto)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval-vision", help="module V: evaluate HSV and YOLO, write results/")
    s.set_defaults(func=cmd_eval_vision)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; the agent speaks Portuguese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
