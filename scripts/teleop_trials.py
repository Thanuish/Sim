"""Teleop trials for any cloth engine: pinch / lift / release, poke and sweep, with both arms.

    python scripts/teleop_trials.py --cloth-engine gpu
    python scripts/teleop_trials.py --cloth-engine mujoco --points 3

Drives the real TeleopSim (clutch mapping, IK, gripper thumbstick, the engine's grasp) in
simulation time, the same for the left and the right arm, and reports per arm:
  pinch      did closing on the fabric pinch it, and does it hold while lifting 10 cm
  stuck      after opening and lifting the gripper away, how much fabric still hangs on it
             (points within 6 cm of the gripper and more than 10 cm above the table: higher than a fold
             of the jeans stands); real: 0
  poke/sweep fabric lifted by a gripper that never pinched (pushed into or swept through it)
and the cost: wall time per simulated second (< 1 = faster than real time).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from vrteleop import scene as S  # noqa: E402
from vrteleop.server import TeleopSim, build_parser  # noqa: E402

CTRL = {"left": np.array([0.0, 0.25, 1.1]), "right": np.array([0.0, -0.25, 1.1])}
TZ = S.TABLE_Z


class Driver:
    """Scripted operator: moves the gripper targets and the thumbsticks in sim time."""

    def __init__(self, server_args):
        args = build_parser().parse_args(["--http", "--no-preview", "--stream-cams", "", "--randomize", "0",
                                          "--data-dir", tempfile.mkdtemp(prefix="teleop-trials-"), *server_args])
        self.sim = TeleopSim(args)
        self.m, self.d, self.cloth = self.sim.m, self.sim.d, self.sim.cloth
        self.ctrl_dt = 1.0 / args.control_hz
        self.wall = 0.0
        self.simt = 0.0

    def reset(self):
        self.sim.reset(randomize=0.0)
        self.home = {s: self.sim.arms.teleop[s].target_pos.copy() for s in S.SIDES}
        self.target = {s: self.home[s].copy() for s in S.SIDES}
        self.stick = {s: 0.0 for s in S.SIDES}

    def advance(self, dur):
        sim, d = self.sim, self.d
        t_end, t0 = d.time + dur, time.perf_counter()
        sim0 = d.time
        while d.time < t_end - 1e-9:
            msg = {"head": [-0.4, 0, 1.65, 1, 0, 0, 0]}
            for s in S.SIDES:
                msg[s] = {"pose": list(CTRL[s] + self.target[s] - self.home[s]) + [1, 0, 0, 0],
                          "grip": 1.0, "stick": [0.0, self.stick[s]], "epoch": 0}
            sim.latest_input, sim.latest_input_t = msg, time.perf_counter()
            sim.control(self.ctrl_dt)
            t_c = d.time + self.ctrl_dt
            while d.time < t_c - 1e-9:
                mujoco.mj_step(self.m, d)
                self.cloth.step(d)
        self.wall += time.perf_counter() - t0
        self.simt += d.time - sim0

    def move(self, side, goal, dur, stick=0.0):
        start = self.target[side].copy()
        n = max(1, int(round(dur / self.ctrl_dt)))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            self.target[side] = start + (np.asarray(goal, float) - start) * a
            self.stick[side] = stick
            self.advance(self.ctrl_dt)
        self.stick[side] = 0.0

    def ee(self, side):
        return self.sim.ee_pose(side)[0]

    def on_gripper(self, side):
        V = self.cloth.verts(self.d)
        near = (np.linalg.norm(V - self.ee(side), axis=1) < 0.06) & (V[:, 2] - TZ > 0.10)
        return int(near.sum())


def spots(V, side, n):
    """n grasp points on this arm's half of the jeans: alternately at an outer edge and mid-leg."""
    half = V[V[:, 1] > 0.05] if side == "left" else V[V[:, 1] < -0.05]
    ys = np.linspace(half[:, 1].min() + 0.06, half[:, 1].max() - 0.06, n)
    out = []
    for k, y in enumerate(ys):
        row = half[np.abs(half[:, 1] - y) < 0.02]
        # outer edge of the far leg, or the middle of that leg (not between the legs: no fabric)
        x = row[:, 0].max() - 0.015 if k % 2 == 0 else row[:, 0].max() - 0.08
        out.append(np.array([x, y]))
    return out


def pinch_trials(dr: Driver, n):
    rows = []
    for where in ("table", "air"):
        for side in S.SIDES:
            for k in range(n):
                dr.reset()
                dr.advance(0.6)
                p = spots(dr.cloth.verts(dr.d), side, n)[k]
                g = np.r_[p, TZ + 0.018]
                dr.move(side, dr.target[side], 0.2)
                dr.move(side, g + [0, 0, 0.08], 1.0)
                dr.move(side, g, 0.8)
                dr.move(side, g, 0.8, stick=+1.0)                 # close
                pinched = len(dr.cloth.held(side))
                dr.move(side, g + [0, 0, 0.10], 1.0)               # lift 10 cm
                V = dr.cloth.verts(dr.d)
                lifted = float(np.sort(V[:, 2])[-20:].mean() - TZ)  # top of the fabric
                still = len(dr.cloth.held(side))
                if where == "table":
                    dr.move(side, g + [0, 0, 0.02], 1.0)           # set it down, open, lift away
                    dr.move(side, dr.target[side], 0.9, stick=-1.0)
                    dr.move(side, g + [0, 0, 0.14], 1.0)
                else:
                    dr.move(side, g + [0, 0, 0.25], 1.0)           # hold it up high, open in the air
                    dr.move(side, dr.target[side], 0.9, stick=-1.0)
                    dr.advance(1.0)                                # it should drop off
                dr.advance(0.5)
                rows.append((where, side, k, pinched, still, lifted, dr.on_gripper(side)))
    return rows


def touch_trials(dr: Driver, n):
    rows = []
    for side in S.SIDES:
        for motion in ("poke", "sweep"):
            for closed in (False, True):
                for k in range(n):
                    dr.reset()
                    dr.advance(0.6)
                    if closed:
                        dr.move(side, dr.target[side], 0.8, stick=+1.0)   # close in the air
                    p = spots(dr.cloth.verts(dr.d), side, n)[k]
                    if motion == "poke":
                        g = np.r_[p, TZ + 0.004]
                        dr.move(side, g + [0, 0, 0.12], 1.0)
                        dr.move(side, g, 1.0)
                        dr.move(side, g + [0.03, 0.02, 0], 0.6)
                        end = g + [0.03, 0.02, 0]
                    else:                                                # across the leg
                        V = dr.cloth.verts(dr.d)
                        row = V[np.abs(V[:, 1] - p[1]) < 0.02]
                        a = np.r_[row[:, 0].max() + 0.05, p[1], TZ + 0.006]
                        b = np.r_[row[:, 0].min() + 0.03, p[1], TZ + 0.006]
                        dr.move(side, a + [0, 0, 0.12], 1.0)
                        dr.move(side, a, 1.0)
                        dr.move(side, b, 1.5)
                        end = b
                    dr.move(side, end + [0, 0, 0.25], 1.2)
                    dr.advance(0.5)
                    rows.append((side, motion, closed, k, len(dr.cloth.held(side)), dr.on_gripper(side)))
    return rows


def replay_human(dr: Driver, path: str, seconds: float | None = None):
    """Feed a recorded operator's controller poses and gripper commands (an episode recorded in
    VR, any engine) into this cloth and report sticking and dropping per arm."""
    import h5py
    h = h5py.File(path, "r")
    fps = float(h.attrs["fps"])
    cp = h["teleop/controller_pose"][:]          # (T, 2, 7) MuJoCo frame
    eng = h["teleop/engaged"][:]
    cmd = h["action_ee/gripper"][:]               # (T, 2)
    head = h["teleop/head_pose"][:]
    T = len(cp) if seconds is None else min(len(cp), int(seconds * fps))
    dr.reset()
    sim, d = dr.sim, dr.d
    rate = sim.arms.teleop["left"].GRIPPER_RATE
    stats = {s: {"held_frames": 0, "stuck_frames": 0, "drops": 0, "pinches": 0, "open_up_frames": 0}
             for s in S.SIDES}
    prev_held = {s: 0 for s in S.SIDES}
    t0, sim0 = time.perf_counter(), d.time
    for k in range(T - 1):
        t_end = d.time + 1.0 / fps
        while d.time < t_end - 1e-9:
            f = k + (d.time - (t_end - 1.0 / fps)) * fps          # fractional frame
            j = min(int(f), T - 2); a = f - j
            msg = {"head": list(np.nan_to_num(head[j], nan=0.0))}
            for si, s in enumerate(S.SIDES):
                pose = cp[j, si] * (1 - a) + cp[j + 1, si] * a
                if not np.all(np.isfinite(pose)):
                    pose = cp[j, si]
                # the stick that reproduces the recorded gripper command
                want = cmd[j + 1, si]
                now = sim.arms.gripper(s)
                stick = float(np.clip((want - now) / (rate * dr.ctrl_dt), -1, 1))
                if abs(stick) < 0.2:
                    stick = 0.0
                msg[s] = {"pose": list(pose) if np.all(np.isfinite(pose)) else None,
                          "grip": 1.0 if eng[j, si] else 0.0, "stick": [0.0, stick], "epoch": 0}
            sim.latest_input, sim.latest_input_t = msg, time.perf_counter()
            sim.control(dr.ctrl_dt)
            t_c = d.time + dr.ctrl_dt
            while d.time < t_c - 1e-9:
                mujoco.mj_step(dr.m, d)
                dr.cloth.step(d)
        for s in S.SIDES:
            st = stats[s]
            n = len(dr.cloth.held(s))
            if n and not prev_held[s]:
                st["pinches"] += 1
            if prev_held[s] and not n and sim.arms.gripper(s) >= 0.9:
                st["drops"] += 1          # lost while commanded closed (not while being opened)
            prev_held[s] = n
            st["held_frames"] += bool(n)
            up = dr.ee(s)[2] - TZ > 0.12
            if not n and up and sim.arms.gripper(s) < 0.3:
                st["open_up_frames"] += 1
                if dr.on_gripper(s) > 20:
                    st["stuck_frames"] += 1
    wall = time.perf_counter() - t0
    print(f"\n  replayed the operator in {Path(path).name}: {T / fps:.0f} s of their hand motions "
          f"(real-time factor {(d.time - sim0) / wall:.2f})")
    for s in S.SIDES:
        st = stats[s]
        print(f"  {s:5s}: pinches {st['pinches']}, dropped while still closed {st['drops']}, "
              f"fabric on the open, raised gripper in {st['stuck_frames']}/{st['open_up_frames']} frames")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=4, help="grasp / touch points per arm")
    ap.add_argument("--only", choices=["pinch", "touch", "all", "none"], default="all")
    ap.add_argument("--human", nargs="*", default=[], metavar="EPISODE",
                    help="also replay these recorded operators' hand motions into the cloth")
    a, server_args = ap.parse_known_args()
    dr = Driver(server_args)
    print(f"cloth engine: {dr.cloth.name}, {dr.cloth.info.nvert} cloth points")
    for ep in a.human:
        replay_human(dr, ep)
    if a.only in ("pinch", "all"):
        rows = pinch_trials(dr, a.points)
        print("\n  pinch / lift 10 cm / then open: set down on the table, or high up in the air")
        print("  open at  arm    spot  pinched  held while lifting  fabric top [mm]  left on open gripper")
        for where, side, k, p, st, lift, stuck in rows:
            print(f"  {where:7s}  {side:5s}  {k:4d}  {p:7d}  {st:18d}  {1e3 * lift:15.0f}  {stuck:8d}")
        for side in S.SIDES:
            r = [x for x in rows if x[1] == side]
            ok = sum(1 for x in r if x[3] > 0 and x[4] > 0 and x[5] > 0.07)
            print(f"  {side}: lifted the fabric in {ok}/{len(r)}, fabric left on the open gripper in "
                  f"{sum(1 for x in r if x[6] > 20)}/{len(r)}")
    if a.only in ("touch", "all"):
        rows = touch_trials(dr, a.points)
        print("\n  poke / sweep without pinching, then lift 25 cm")
        for side in S.SIDES:
            r = [x for x in rows if x[0] == side]
            bad = [(x[1], 'closed' if x[2] else 'open', x[3], x[5]) for x in r if x[5] > 20 or x[4] > 0]
            print(f"  {side}: fabric on the gripper (or pinched by accident) in {len(bad)}/{len(r)} {bad[:6]}")
    print(f"\n  cost: {dr.wall / max(dr.simt, 1e-9):.2f} s wall per simulated second "
          f"(real-time factor {dr.simt / max(dr.wall, 1e-9):.2f})")


if __name__ == "__main__":
    main()
