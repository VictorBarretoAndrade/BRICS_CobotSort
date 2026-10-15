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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
