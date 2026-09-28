"""MuJoCo dual-UR5e teleoperation server for Meta Quest (WebXR).

    python -m vrteleop.server                # jeans folding task, HTTPS on :8443 (self-signed)
    python -m vrteleop.server --task blocks  # pick-and-place blocks into a bin
    python -m vrteleop.server --cameras      # also record camera images
    python -m vrteleop.server --http         # plain HTTP on :8080 (use with `adb reverse`)

Open the printed URL in the Quest browser, press "Enter VR".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import ssl
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import WSMsgType, web

from . import cloth as C
from . import scene as S
from .ik import DiffIK
from .pipeline import Session
from .recorder import EpisodeRecorder
from .teleop import ArmTeleop
from .webscene import export_scene

WEB_DIR = Path(__file__).resolve().parent / "web"
GRIPPER_JOINT_MAX = 0.8     # 2F-85 driver joint range [0, 0.8] rad
PROTOCOL_VERSION = 5        # bump when server <-> web client messages change


class TeleopSim:
    def __init__(self, args):
        self.args = args
        self.info = S.build_scene(args.task, args.cloth_spacing)
        self.m = self.info.model
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(args.seed)
        self.ik = {s: DiffIK(self.m, self.info.arm_qpos_adr[s], self.info.arm_dof_adr[s],
                             self.info.ee_site[s], S.HOME_Q[s]) for s in S.SIDES}
        self.teleop = {s: ArmTeleop(pos_scale=args.scale, z_min=S.PINCH_Z_MIN[args.task]) for s in S.SIDES}
        self.q_cmd = {s: S.HOME_Q[s].copy() for s in S.SIDES}
        self.cloth = self.info.cloth
        self.grasp = {}
        if self.cloth is not None:
            self.grasp = {s: C.PinchGrasp(self.m, self.cloth, s, self.info.grasp_eq[s],
                                          slip_force=args.slip_force) for s in S.SIDES}
        self.cloth_metrics = {}
        self._metrics_t = 0.0
        self.render_bodies, self.scene_gz = export_scene(self.m, cloth=self.cloth,
                                                         web_textures=self.info.web_textures)
        print(f"[scene] task '{args.task}': {S.TASKS[args.task]}"
              + (f"  (jeans: {self.cloth.nvert} cloth vertices)" if self.cloth is not None else ""))
        print(f"[scene] {len(self.render_bodies)} bodies, web payload {len(self.scene_gz) / 1e6:.1f} MB (gzip)")

        self.latest_input: dict = {}
        self.latest_input_t = 0.0
        self.input_count = 0
        self.input_hz = 0.0
        self._input_log_t = time.perf_counter()
        self.clients: set[web.WebSocketResponse] = set()

        cams = []
        self.renderer = None           # one offscreen renderer shared by recording + VR streaming
        if args.cameras:
            cams = [c for c in (args.camera_names.split(",") if args.camera_names else self.info.cameras)]
        # Live camera feeds shown as floating screens in VR (toggle: left stick click)
        self.stream_cams = [c for c in args.stream_cams.split(",") if c] if args.stream_cams else []
        self.cams_on = bool(self.stream_cams)
        self.session = Session(
            Path(args.data_dir), args.task, name=args.session or None, operator=args.operator,
            preview=not args.no_preview,
            settings={k: v for k, v in vars(args).items() if k not in ("host",)})
        print(f"[data] recording to {self.session.dir}")
        self.recorder = EpisodeRecorder(
            self.session.dir, args.fps, self.info.xml, list(self.info.objects), self._joint_names(), cams,
            task=args.task,
            cloth_faces=self.cloth.faces if self.cloth is not None else None,
            cloth_rest_uv=self.cloth.rest_uv if self.cloth is not None else None)
        self.last_saved = None
        self.message = ""
        self.reset(randomize=0.0)

    # ------------------------------------------------------------------ state
    def _joint_names(self):
        names = []
        for s in S.SIDES:
            names += [f"{s}_{j}" for j in S.ARM_JOINTS] + [f"{s}_gripper"]
        return names

    def reset(self, randomize: float | None = None):
        r = self.args.randomize if randomize is None else randomize
        S.reset(self.info, self.d, self.rng, randomize=r, cloth_init=self.args.cloth_init)
        for g in self.grasp.values():
            g.reset(self.m, self.d)
        self.cloth_metrics = {}
        for s in S.SIDES:
            self.q_cmd[s] = S.HOME_Q[s].copy()
            pos, quat = self.ik[s].fk(self.q_cmd[s], self.d.qpos)
            self.teleop[s].reset(pos, quat)

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, self.args.img_h, self.args.img_w)
        return self.renderer

    def camera_packet(self, i: int) -> bytes:
        """Render stream camera i and JPEG-encode it: [2, cam_index, 0, 0] + jpeg."""
        from io import BytesIO
        from PIL import Image
        r = self.get_renderer()
        r.update_scene(self.d, camera=self.stream_cams[i])
        buf = BytesIO()
        Image.fromarray(r.render()).save(buf, "JPEG", quality=self.args.jpeg_quality)
        return bytes([2, i, 0, 0]) + buf.getvalue()

    def ee_pose(self, s):
        sid = self.info.ee_site[s]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[sid])
        return self.d.site_xpos[sid].copy(), q

    # ---------------------------------------------------------------- control
    def control(self, dt: float):
        fresh = (time.perf_counter() - self.latest_input_t) < self.args.input_timeout
        for s in S.SIDES:
            tel = self.teleop[s]
            if not tel.engaged:
                # keep the virtual target glued to the commanded pose while idle
                tel.target_pos, tel.target_quat = self.ik[s].fk(self.q_cmd[s])
            if fresh:
                tel.update(self.latest_input.get(s), dt)
            else:
                tel.disengage()
            if tel.engaged:
                self.q_cmd[s] = self.ik[s].step(self.q_cmd[s], tel.target_pos, tel.target_quat, dt)
            self.d.ctrl[self.info.arm_act[s]] = self.q_cmd[s]
            self.d.ctrl[self.info.grip_act[s]] = 255.0 * tel.gripper
            if s in self.grasp:
                self.grasp[s].update(self.m, self.d, tel.gripper, dt)

    def update_cloth_metrics(self):
        if self.cloth is not None:
            self.cloth_metrics = C.fold_metrics(self.cloth, C.verts(self.cloth, self.d), S.TABLE_Z)

    # -------------------------------------------------------------- recording
    def record_frame(self):
        d, info = self.d, self.info
        qpos, qvel, act = [], [], []
        ee_p, ee_q, tp, tq, gc, eng, cp = [], [], [], [], [], [], []
        for s in S.SIDES:
            qa, va = info.arm_qpos_adr[s], info.arm_dof_adr[s]
            ga = info.grip_qpos_adr[s]
            gv = self.m.jnt_dofadr[self.m.joint(f"{s}_gripper_right_driver_joint").id]
            qpos += list(d.qpos[qa]) + [d.qpos[ga] / GRIPPER_JOINT_MAX]
            qvel += list(d.qvel[va]) + [d.qvel[gv] / GRIPPER_JOINT_MAX]
            tel = self.teleop[s]
            act += list(self.q_cmd[s]) + [tel.gripper]
            p, q = self.ee_pose(s)
            ee_p.append(p); ee_q.append(q)
            tp.append(tel.target_pos); tq.append(tel.target_quat); gc.append(tel.gripper)
            eng.append(tel.engaged)
            c = self.latest_input.get(s) or {}
            cp.append(c.get("pose") or [np.nan] * 7)
        objs = []
        for n in info.objects:
            a = info.obj_qpos_adr[n]
            objs.append(d.qpos[a:a + 7].copy())
        frame = dict(time=d.time, qpos=qpos, qvel=qvel, ee_pos=ee_p, ee_quat=ee_q,
                     object_pose=objs, full_qpos=d.qpos.copy(), full_qvel=d.qvel.copy(),
                     action=act, target_pos=tp, target_quat=tq, gripper_cmd=gc, engaged=eng,
                     head_pose=self.latest_input.get("head") or [np.nan] * 7,
                     controller_pose=cp)
        if self.cloth is not None:
            frame["cloth_verts"] = C.verts(self.cloth, d).copy()
            frame["cloth_grasp"] = [(self.grasp[s].held + [-1] * C.PinchGrasp.K)[:C.PinchGrasp.K]
                                    for s in S.SIDES]
        if self.recorder.camera_names:
            r = self.get_renderer()
            for cam in self.recorder.camera_names:
                r.update_scene(d, camera=cam)
                frame[f"img_{cam}"] = r.render().copy()
        self.recorder.add(frame)

    def log_input(self):
        """Print a one-line summary of what the headset is sending (helps debug controls)."""
        parts = [f"[input] {self.input_hz:4.0f} Hz"]
        for s in S.SIDES:
            c = self.latest_input.get(s)
            if not c:
                parts.append(f"{s[0].upper()}: no controller")
                continue
            st = c.get("stick") or [0, 0]
            pressed = [i for i, b in enumerate(c.get("buttons") or []) if b]
            parts.append(f"{s[0].upper()}: grip {c.get('grip', 0):.2f} stick ({st[0]:+.2f},{st[1]:+.2f}) "
                         f"btn {pressed} {'ENGAGED' if self.teleop[s].engaged else ''} "
                         f"grip% {self.teleop[s].gripper * 100:.0f}")
        print("  ".join(parts), flush=True)

    def command(self, cmd: str):
        rec = self.recorder
        if cmd == "record_toggle":
            if rec.active:
                self.save_episode(success=True)
            else:
                rec.start(self.d.time)
                self.message = f"Recording episode {rec.next_index()}"
        elif cmd == "save_fail":
            if rec.active:
                self.save_episode(success=False)
        elif cmd == "discard":
            if rec.active:
                rec.discard()
                self.message = "Episode discarded"
            self.reset()
        elif cmd == "cams_toggle":
            self.cams_on = not self.cams_on and bool(self.stream_cams)
            self.message = f"Camera screens {'on' if self.cams_on else 'off'}"
        elif cmd == "reset":
            if rec.active:
                rec.discard()
                self.message = "Episode discarded (reset)"
            else:
                self.message = "Scene reset"
            self.reset()
        print(f"[cmd] {cmd}: {self.message}")

    def save_episode(self, success: bool):
        extra = {}
        if self.cloth is not None:
            self.update_cloth_metrics()
            extra = {"final_coverage": self.cloth_metrics["coverage"],
                     "final_height": self.cloth_metrics["height"],
                     "cloth_spacing": self.cloth.garment.spacing}
        n_frames = self.recorder.num_frames
        path = self.recorder.stop(self.m.opt.timestep, success=success, extra_attrs=extra)
        if path:
            self.session.episode_saved(path, {
                "success": bool(success),
                "duration_s": round(n_frames / self.args.fps, 2),
                "frames": n_frames,
                "final_coverage": round(extra["final_coverage"], 3) if extra else "",
                "final_height_m": round(extra["final_height"], 3) if extra else "",
                "notes": "",
            })
            print(f"[data] saved {self.session.relpath(path)}")
            self.last_saved = path.name
            self.message = f"Saved {path.name} ({'success' if success else 'failure'})"
        else:
            self.message = "Episode too short, not saved"
        self.reset()

    # -------------------------------------------------------------- streaming
    def pose_packet(self) -> bytes:
        d = self.d
        flags = (1 if self.teleop["left"].engaged else 0) | (2 if self.teleop["right"].engaged else 0) \
            | (4 if self.recorder.active else 0)
        rb = self.render_bodies
        poses = np.concatenate([d.xpos[rb], d.xquat[rb]], axis=1).astype(np.float32)
        tgt = np.array([np.concatenate([self.teleop[s].target_pos, self.teleop[s].target_quat])
                        for s in S.SIDES], dtype=np.float32)
        head = self.latest_input.get("head")
        head_live = head is not None and (time.perf_counter() - self.latest_input_t) < 1.0
        if head_live:
            flags |= 8
        head_arr = np.asarray(head if head_live else [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        cloth = (C.verts(self.cloth, d).astype(np.float32) if self.cloth is not None
                 else np.zeros((0, 3), np.float32))
        header = np.array([d.time, flags, len(rb), len(cloth)], dtype=np.float32)
        # layout: header(4) | bodies(n*7) | targets(2*7) | operator head pose(7) | cloth verts(N*3)
        # (all in the MuJoCo frame)
        return (bytes([1, 0, 0, 0]) + header.tobytes() + poses.tobytes() + tgt.tobytes()
                + head_arr.tobytes() + cloth.tobytes())

    def status(self, rtf: float) -> str:
        return json.dumps({
            "type": "status",
            "version": PROTOCOL_VERSION,
            "task": self.args.task,
            "task_title": S.TASKS[self.args.task],
            "holding": {s: g.holding for s, g in self.grasp.items()},
            "cloth": {k: round(float(v), 3) for k, v in self.cloth_metrics.items()},
            "recording": self.recorder.active,
            "frames": self.recorder.num_frames,
            "rec_time": self.recorder.num_frames / self.args.fps,
            "next_episode": self.recorder.next_index(),
            "last_saved": self.last_saved,
            "message": self.message,
            "sim_time": round(self.d.time, 2),
            "rtf": round(rtf, 2),
            "engaged": {s: self.teleop[s].engaged for s in S.SIDES},
            "gripper": {s: round(self.teleop[s].gripper, 2) for s in S.SIDES},
            "cams": self.stream_cams,
            "cams_on": self.cams_on,
            "input_fresh": (time.perf_counter() - self.latest_input_t) < self.args.input_timeout,
            "input_hz": round(self.input_hz, 1),
        })

    async def broadcast(self, data, binary=True):
        dead = []
        for ws in self.clients:
            try:
                if binary:
                    await ws.send_bytes(data)
                else:
                    await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def run(self):
        m, d = self.m, self.d
        ctrl_dt = 1.0 / self.args.control_hz
        pub_dt = 1.0 / self.args.stream_hz
        rec_dt = 1.0 / self.args.fps
        wall0, sim0 = time.perf_counter(), d.time
        next_ctrl = d.time
        next_rec = d.time
        next_pub = next_status = next_cam = 0.0
        cam_dt = 1.0 / max(self.args.cam_stream_hz, 0.1)
        cam_i = 0
        rtf_w, rtf_s, rtf = time.perf_counter(), d.time, 1.0
        while True:
            now = time.perf_counter()
            target = sim0 + (now - wall0)
            steps = 0
            while d.time < target and steps < 25:
                if d.time >= next_ctrl - 1e-9:
                    self.control(ctrl_dt)
                    next_ctrl = d.time + ctrl_dt
                if self.recorder.active and d.time >= next_rec - 1e-9:
                    self.record_frame()
                    next_rec = d.time + rec_dt
                elif not self.recorder.active:
                    next_rec = d.time
                mujoco.mj_step(m, d)
                steps += 1
                if d.time < sim0:            # reset() rewound the clock
                    break
            behind = d.time < target - 0.004       # physics has not caught up with the wall clock
            if d.time < target - 0.1 or d.time < sim0:   # fell behind / reset: resync clock
                wall0, sim0 = time.perf_counter(), d.time
                next_ctrl = next_rec = d.time
            if now >= next_pub and self.clients:
                await self.broadcast(self.pose_packet())
                next_pub = now + pub_dt
            if self.cams_on and self.clients and now >= next_cam and not behind:
                # Physics first: camera screens are only rendered while the sim keeps real time.
                # One camera per loop iteration (round-robin) so physics is never starved.
                # If rendering is slow, the stream rate drops automatically (<= ~20% of wall time).
                t_r = time.perf_counter()
                await self.broadcast(self.camera_packet(cam_i))
                cam_i = (cam_i + 1) % len(self.stream_cams)
                spent = time.perf_counter() - t_r
                next_cam = now + max(cam_dt / len(self.stream_cams), 4.0 * spent)
            if now - self._input_log_t >= 2.0:
                self.input_hz = self.input_count / (now - self._input_log_t)
                self.input_count = 0
                self._input_log_t = now
                if self.input_hz > 0:
                    self.log_input()
            since = now - self._metrics_t       # fold score: every 1 s when idle time allows, 3 s at most
            if self.cloth is not None and (since > 3.0 or (since > 1.0 and not behind)):
                self.update_cloth_metrics()
                self._metrics_t = time.perf_counter()
            if now >= next_status:
                if now - rtf_w > 0.5:
                    rtf = (d.time - rtf_s) / (now - rtf_w) if d.time >= rtf_s else 1.0
                    rtf_w, rtf_s = now, d.time
                if self.clients:
                    await self.broadcast(self.status(rtf), binary=False)
                next_status = now + 0.2
            await asyncio.sleep(0.001)


# ---------------------------------------------------------------------- web
def make_app(sim: TeleopSim) -> web.Application:
    @web.middleware
    async def no_cache_app_js(request, handler):
        resp = await handler(request)
        if request.path.startswith("/static/") and "/vendor/" not in request.path:
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app = web.Application(middlewares=[no_cache_app_js])

    boot_id = str(int(time.time()))
    index_html = (WEB_DIR / "index.html").read_text().replace(
        "/static/main.js", f"/static/main.js?v={boot_id}")   # never run a stale cached client

    async def index(_):
        return web.Response(text=index_html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def scene_json(_):
        return web.Response(body=sim.scene_gz, content_type="application/json",
                            headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
        await ws.prepare(request)
        sim.clients.add(ws)
        peer = request.remote
        print(f"[ws] client connected: {peer} ({len(sim.clients)} total)")
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                t = data.get("type")
                if t == "input":
                    sim.latest_input = data
                    sim.latest_input_t = time.perf_counter()
                    sim.input_count += 1
                elif t == "cmd":
                    sim.command(data.get("cmd", ""))
        finally:
            sim.clients.discard(ws)
            print(f"[ws] client disconnected: {peer}")
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", WEB_DIR, show_index=False)

    async def start_sim(app_):
        app_["sim_task"] = asyncio.create_task(sim.run())

    async def stop_sim(app_):
        app_["sim_task"].cancel()
        if sim.recorder.active:
            sim.save_episode(success=False)

    app.on_startup.append(start_sim)
    app.on_cleanup.append(stop_sim)
    return app


def lan_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=None, help="default 8443 (https) / 8080 (http)")
    p.add_argument("--http", action="store_true", help="serve plain HTTP (for adb reverse / localhost)")
    p.add_argument("--data-dir", default=str(S.ROOT / "data"),
                   help="root folder; episodes go to <data-dir>/<task>/<session>/")
    p.add_argument("--session", default="",
                   help="session folder name (default: date and time); reuse a name to add to it")
    p.add_argument("--operator", default=os.environ.get("USER") or os.environ.get("USERNAME", ""),
                   help="stored with every episode")
    p.add_argument("--no-preview", action="store_true", help="don't render an MP4 preview per episode")
    p.add_argument("--fps", type=float, default=30.0, help="recording rate (sim time)")
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--stream-hz", type=float, default=90.0)
    p.add_argument("--scale", type=float, default=1.0, help="hand->robot translation scale")
    p.add_argument("--task", choices=list(S.TASKS), default=S.DEFAULT_TASK,
                   help="jeans: fold a pair of jeans (cloth); blocks: pick-and-place into a bin")
    p.add_argument("--cloth-init", choices=["flat", "crumpled"], default="flat",
                   help="jeans task: start spread flat, or dropped into a random heap")
    p.add_argument("--cloth-spacing", type=float, default=None,
                   help="jeans task: cloth simulation resolution [m] (default 0.05; 0.04 is finer but "
                        "needs a fast CPU to stay real time)")
    p.add_argument("--slip-force", type=float, default=30.0,
                   help="jeans task: pull [N] at which fabric slips out of a pinch")
    p.add_argument("--randomize", type=float, default=0.05,
                   help="object xy jitter on reset [m] (jeans: also +/- 3x this in heading [rad])")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cameras", action="store_true", help="record camera images into episodes")
    p.add_argument("--camera-names", default="", help="comma list; default = all cameras")
    p.add_argument("--img-w", type=int, default=320)
    p.add_argument("--img-h", type=int, default=240)
    p.add_argument("--stream-cams", default="left_gripper_wrist_cam,head_cam,right_gripper_wrist_cam",
                   help="cameras shown as live screens in VR (comma list, '' to disable)")
    p.add_argument("--cam-stream-hz", type=float, default=15.0)
    p.add_argument("--input-timeout", type=float, default=0.35,
                   help="disengage arms if no controller data for this long [s]")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--public-host", default=os.environ.get("TELEOP_PUBLIC_HOST", ""),
                   help="address(es) the headset uses to reach this machine, comma separated "
                        "(needed in Docker, where the container can't see the host's IP)")
    args = p.parse_args(argv)

    if not S.UR5E_XML.exists() or not S.GRIPPER_XML.exists():
        sys.exit("Robot assets missing. Run:  python setup_assets.py")

    sim = TeleopSim(args)
    app = make_app(sim)
    public = [h.strip() for h in args.public_host.split(",") if h.strip()]
    ips = public + ["127.0.0.1"] if public else lan_ips()
    ssl_ctx = None
    if not args.http:
        from .certs import ensure_cert
        cert, key = ensure_cert(S.ROOT / "certs", ips)
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(cert, key)
    port = args.port or (8080 if args.http else 8443)
    scheme = "http" if args.http else "https"
    print("\n  Open on your Quest browser:")
    for ip in ips:
        if ip != "127.0.0.1":
            print(f"    {scheme}://{ip}:{port}")
    print(f"    {scheme}://localhost:{port}   (desktop / adb reverse)\n")
    if not args.http:
        print("  First visit: accept the self-signed certificate warning (Advanced -> Proceed).\n")
    web.run_app(app, host=args.host, port=port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    main()
