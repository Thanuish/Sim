"""Jeans on the GPU: XPBD cloth (extended position-based dynamics) written in Taichi.

Runs on any Vulkan GPU (AMD, NVIDIA, Intel). The garment is the same two-panel pair of jeans as
the MuJoCo cloth (cloth.jeans_garment: front and back panel sewn along the out- and inseams,
open at the waist and the hems), but at a much finer resolution (~1 cm instead of 5.5 cm).

Per substep (Macklin et al. 2016, "XPBD"; Müller 2020, small steps):
  * predict positions from velocities, gravity and air drag (speed capped so that nothing moves
    more than a fraction of the cloth thickness per substep: no tunnelling); pinned points
    follow their gripper (or a world target);
  * threads (warp/weft edges) resist stretching, bias diagonals shear softly, and a bending
    constraint across every interior edge keeps folds round;
  * self-collision: every point keeps `thickness` from every other point that isn't its
    neighbour on the same panel of the flat pattern (uniform-grid spatial hash);
  * the robot: its collision geoms as oriented boxes, moved smoothly between their poses at
    the last two frames, push the fabric out (with friction); the table likewise.
Constraints are solved in parallel (Jacobi with averaging), which suits the GPU.
"""
# (no `from __future__ import annotations`: Taichi reads the kernels' type hints)
import numpy as np
import taichi as ti

from .cloth import GarmentSpec, fine_garment, place_flat
from .cloth_config import GPUSettings

_initialised = None
FREE, WORLD = -1, 2          # pin modes; 0 / 1 = pinned to gripper 0 / 1 (left / right)


def init(arch: str = "gpu"):
    """Start Taichi once per process. arch: gpu (the best available: CUDA, Vulkan or Metal),
    vulkan, cuda, metal or cpu. Raises RuntimeError if a GPU was asked for and there is none
    (Taichi would quietly run on the CPU, far too slowly for real time)."""
    global _initialised
    if _initialised is None:
        ti.init(arch=getattr(ti, arch), log_level=ti.WARN, random_seed=0)
        got = ti.lang.impl.current_cfg().arch
        _initialised = str(got).split(".")[-1]
        if arch != "cpu" and got in (ti.cpu, ti.x64, ti.arm64):
            raise RuntimeError(f"no GPU for the cloth (Taichi found no {arch} backend)")
    elif arch != "cpu" and _initialised in ("x64", "arm64", "cpu"):
        raise RuntimeError("no GPU for the cloth")
    return _initialised


class GPUJeans:
    """A pair of jeans lying flat at `pos` (waistband centre), legs along `yaw`, with up to
    `max_boxes` robot colliders."""

    HASH_SIZE = 1 << 17
    BUCKET = 24

    def __init__(self, garment: GarmentSpec, params: GPUSettings, pos, yaw: float, table_z: float,
                 max_boxes: int = 64):
        self.p = p = params
        g, J, self.thick = fine_garment(garment, p.spacing, p.thickness)
        self.J, self.garment = J, g
        self.faces = J["faces"].astype(np.int32)
        X0 = place_flat(J, pos, yaw, table_z, 0.5 * self.thick)
        self.X0 = X0
        self.N = N = len(X0)
        self.table_z = table_z
        # constraints: threads (stretch), bias (shear), bending (across each interior edge)
        cons, comp = [], []
        for e in J["threads"]:
            cons.append(e); comp.append(p.stretch_compliance)
        for e in J["diagonals"]:
            cons.append(e); comp.append(p.shear_compliance)
        for a, b in _bend_pairs(self.faces):
            cons.append((a, b)); comp.append(p.bend_compliance)
        cons = np.array(cons, np.int32)
        self.n_cons = len(cons)
        rest = np.linalg.norm(X0[cons[:, 0]] - X0[cons[:, 1]], axis=1).astype(np.float32)

        V3 = lambda n: ti.Vector.field(3, ti.f32, n)
        self.x, self.xp, self.v, self.target, self.dx = V3(N), V3(N), V3(N), V3(N), V3(N)
        self.cnt = ti.field(ti.f32, N)
        self.invm = ti.field(ti.f32, N)
        # pattern position and panel (0 front, 1 back, 2 seam) of every point: two points skip
        # self-collision only if they are neighbours on the same panel of the flat pattern
        self.uv = ti.Vector.field(2, ti.f32, N)
        self.panel = ti.field(ti.i32, N)
        self.pin = ti.field(ti.i32, N)                 # FREE, WORLD or gripper index
        self.pin_local = V3(N)                         # position in the gripper frame
        self.c_ij = ti.Vector.field(2, ti.i32, len(cons))
        self.c_rest = ti.field(ti.f32, len(cons))
        self.c_comp = ti.field(ti.f32, len(cons))
        self.cell_n = ti.field(ti.i32, self.HASH_SIZE)
        self.cell_p = ti.field(ti.i32, (self.HASH_SIZE, self.BUCKET))
        # robot colliders: oriented boxes at the previous and the current frame
        self.max_boxes = max_boxes
        self.n_boxes = ti.field(ti.i32, ())
        self.b_c0, self.b_c1, self.b_half = V3(max_boxes), V3(max_boxes), V3(max_boxes)
        self.b_R0 = ti.Matrix.field(3, 3, ti.f32, max_boxes)
        self.b_R1 = ti.Matrix.field(3, 3, ti.f32, max_boxes)
        self.b_side = ti.field(ti.i32, max_boxes)      # the arm (0 / 1) a box belongs to, or -1
        self.b_side.fill(-1)
        # friction of arm k against the fabric (set_arm_friction)
        self.arm_fr = ti.field(ti.f32, 2)
        # long-range attachments (Kim et al. 2012): per point and gripper, the nearest pinched
        # point and the most it may be away from it (the fabric between them, unstretched)
        self.teth_j = ti.field(ti.i32, (N, 2))
        self.teth_L = ti.field(ti.f32, (N, 2))
        # gripper frames (for pinned points), previous and current frame
        self.g_c0, self.g_c1 = V3(2), V3(2)
        self.g_R0 = ti.Matrix.field(3, 3, ti.f32, 2)
        self.g_R1 = ti.Matrix.field(3, 3, ti.f32, 2)
        self.alpha = ti.field(ti.f32, ())              # substep progress through the frame
        # all robot poses of a frame go up in one transfer (each from_numpy costs ~0.25 ms)
        self.BOX_F, self.GRIP_F = 27, 24               # floats per box / gripper in pose_buf
        self.pose_buf = ti.field(ti.f32, max_boxes * self.BOX_F + 2 * self.GRIP_F)

        self.m = garment.mass / N
        self.uv.from_numpy(J["uv"].astype(np.float32))
        self._panel_np = J["layer"].astype(np.int32)
        self.panel.from_numpy(self._panel_np)
        self.c_ij.from_numpy(cons); self.c_rest.from_numpy(rest)
        self.c_comp.from_numpy(np.array(comp, np.float32))
        eye = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        self.g_R0.from_numpy(eye); self.g_R1.from_numpy(eye)
        self.dt = 1.0 / p.fps / p.substeps
        self._build_kernels()
        self.reset()

    # ---------------------------------------------------------------- kernels
    def _build_kernels(self):
        x, xp, v, uv, panel, invm, target, dx, cnt = (self.x, self.xp, self.v, self.uv, self.panel,
                                                      self.invm, self.target, self.dx, self.cnt)
        pin, pin_local = self.pin, self.pin_local
        c_ij, c_rest, c_comp, cell_n, cell_p = self.c_ij, self.c_rest, self.c_comp, self.cell_n, self.cell_p
        n_boxes, b_c0, b_c1, b_R0, b_R1, b_half = (self.n_boxes, self.b_c0, self.b_c1, self.b_R0,
                                                   self.b_R1, self.b_half)
        g_c0, g_c1, g_R0, g_R1, alpha = self.g_c0, self.g_c1, self.g_R0, self.g_R1, self.alpha
        b_side, arm_fr = self.b_side, self.arm_fr
        teth_j, teth_L = self.teth_j, self.teth_L
        DT, TH, HS, BK = self.dt, self.thick, self.HASH_SIZE, self.BUCKET
        MAXMOVE, DRAG, W = 0.4 * self.thick, self.p.air_drag, self.p.relaxation
        TZ, FR, GFR = self.table_z + 0.5 * self.thick, self.p.table_friction, self.p.gripper_friction
        CFR = self.p.cloth_friction
        R = 0.5 * self.thick                           # point radius against the robot and table
        DA = 1.0 / self.p.substeps                     # substep, as a fraction of the frame

        @ti.func
        def hash_of(c):
            return ((c[0] * 73856093) ^ (c[1] * 19349663) ^ (c[2] * 83492791)) & (HS - 1)

        @ti.func
        def lerp_frame(c0, c1, R0, R1, a):
            """Pose between two frames: positions lerped, rotation lerped and re-orthonormalised."""
            c = c0 + (c1 - c0) * a
            M = R0 + (R1 - R0) * a
            e0 = ti.Vector([M[0, 0], M[1, 0], M[2, 0]]).normalized()
            e1 = ti.Vector([M[0, 1], M[1, 1], M[2, 1]])
            e1 = (e1 - e0.dot(e1) * e0).normalized()
            e2 = e0.cross(e1)
            return c, ti.Matrix.cols([e0, e1, e2])

        # predict / solve_* are inlined into one kernel per substep (substep below): the GPU still
        # runs their loops one after the other, but Python launches 1 kernel instead of 4
        @ti.func
        def predict(a: ti.f32):
            for i in x:
                xp[i] = x[i]
                if pin[i] == FREE:
                    v[i] += ti.Vector([0.0, 0.0, -9.81]) * DT
                    v[i] *= 1.0 - DRAG * DT
                    sp = v[i].norm() * DT
                    if sp > MAXMOVE:
                        v[i] *= MAXMOVE / sp
                    x[i] += v[i] * DT
                elif pin[i] == WORLD:
                    x[i] = target[i]
                else:
                    k = pin[i]
                    c, Rm = lerp_frame(g_c0[k], g_c1[k], g_R0[k], g_R1[k], a)
                    x[i] = c + Rm @ pin_local[i]

        @ti.kernel
        def build_hash():
            for c in cell_n:
                cell_n[c] = 0
            for i in x:
                h = hash_of(ti.floor(x[i] / TH, ti.i32))
                k = ti.atomic_add(cell_n[h], 1)
                if k < BK:
                    cell_p[h, k] = i

        @ti.func
        def solve_tethers():
            """A point never gets further from a pinched point than the fabric between them:
            denim hardly stretches, and the threads alone (Jacobi, one pass per substep) let a
            lifted leg stretch like rubber and snap back when let go."""
            for i in x:
                if invm[i] > 0:
                    for k in ti.static(range(2)):
                        j = teth_j[i, k]
                        if j >= 0:
                            d = x[i] - x[j]
                            l = d.norm()
                            if l > teth_L[i, k]:
                                x[i] -= (l - teth_L[i, k]) / l * d

        @ti.func
        def solve_constraints():
            for i in x:
                dx[i] = ti.Vector([0.0, 0.0, 0.0]); cnt[i] = 0.0
            for e in c_ij:
                i, j = c_ij[e][0], c_ij[e][1]
                w = invm[i] + invm[j]
                if w > 0:
                    d = x[i] - x[j]
                    l = d.norm() + 1e-9
                    lam = -(l - c_rest[e]) / (w + c_comp[e] / (DT * DT))
                    n = d / l
                    dx[i] += lam * invm[i] * n
                    dx[j] -= lam * invm[j] * n
                    cnt[i] += 1.0; cnt[j] += 1.0
            for i in x:
                if cnt[i] > 0:
                    x[i] += W * dx[i] / cnt[i]

        @ti.func
        def solve_self():
            for i in x:
                dx[i] = ti.Vector([0.0, 0.0, 0.0]); cnt[i] = 0.0
            for i in x:
                c = ti.floor(x[i] / TH, ti.i32)
                for a, b, cc in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    h = hash_of(c + ti.Vector([a, b, cc]))
                    for k in range(ti.min(cell_n[h], BK)):
                        j = cell_p[h, k]
                        if j != i:
                            d = x[i] - x[j]
                            l = d.norm()
                            same = panel[i] == panel[j] or panel[i] == 2 or panel[j] == 2
                            neighbours = same and (uv[i] - uv[j]).norm() < 1.5 * TH
                            if l < TH and l > 1e-9 and not neighbours:
                                # a pinned partner doesn't move: the free point takes it all
                                share = 0.5 if invm[j] > 0 else 1.0
                                n = d / l
                                pen = TH - l
                                # fabric on fabric: Coulomb friction cancels the sliding between
                                # the two points, at most mu x the penetration (folds stay put)
                                rel = (x[i] - xp[i]) - (x[j] - xp[j])
                                rt = rel - rel.dot(n) * n
                                rl = rt.norm()
                                if rl > CFR * pen:
                                    rt *= CFR * pen / rl
                                dx[i] += share * (pen * n - rt)
                                cnt[i] += 1.0
            for i in x:
                if cnt[i] > 0 and invm[i] > 0:
                    x[i] += dx[i] / cnt[i]

        @ti.func
        def solve_contacts(a: ti.f32):
            nb = n_boxes[None]
            for i in x:
                if invm[i] > 0:
                    for b in range(nb):
                        c, Rm = lerp_frame(b_c0[b], b_c1[b], b_R0[b], b_R1[b], a)
                        lp = Rm.transpose() @ (x[i] - c)
                        h = b_half[b]
                        hx = h + R
                        if ti.abs(lp[0]) < hx[0] and ti.abs(lp[1]) < hx[1] and ti.abs(lp[2]) < hx[2]:
                            # sphere (the point, radius R) against the box: push out along the
                            # direction from the closest point of the box, which is rounded at
                            # edges and corners, so fabric slides off them instead of catching
                            q = ti.max(ti.min(lp, h), -h)
                            dv = lp - q
                            dist = dv.norm()
                            lq = lp
                            nl = ti.Vector([0.0, 0.0, 0.0])
                            touching = True
                            if dist > 1e-7:
                                if dist < R:
                                    nl = dv / dist
                                    lq = q + nl * R
                                else:
                                    touching = False      # in the corner region but clear
                            else:
                                # centre inside the box: out through the nearest face
                                pen = h - ti.abs(lp)
                                if pen[0] <= pen[1] and pen[0] <= pen[2]:
                                    nl[0] = ti.select(lp[0] > 0, 1.0, -1.0)
                                elif pen[1] <= pen[2]:
                                    nl[1] = ti.select(lp[1] > 0, 1.0, -1.0)
                                else:
                                    nl[2] = ti.select(lp[2] > 0, 1.0, -1.0)
                                lq = lp + nl * (pen.min() + R)
                            if not touching:
                                continue
                            # Coulomb friction (position based): the sliding relative to the box,
                            # which moved since the last substep, is cancelled at most up to
                            # mu x the penetration, so fabric only brushing a finger isn't carried
                            cp, Rp = lerp_frame(b_c0[b], b_c1[b], b_R0[b], b_R1[b], ti.max(a - DA, 0.0))
                            new = c + Rm @ lq
                            depth = (new - x[i]).norm()
                            carried = new - (cp + Rp @ lq)
                            slide = (new - xp[i]) - carried
                            n = Rm @ nl
                            slide -= slide.dot(n) * n
                            sl = slide.norm()
                            mu = GFR
                            if b_side[b] >= 0:
                                mu = arm_fr[b_side[b]]
                            if sl > mu * depth:
                                slide *= mu * depth / sl
                            x[i] = new - slide
                    if x[i][2] < TZ:
                        depth = TZ - x[i][2]
                        x[i][2] = TZ
                        t = x[i] - xp[i]
                        t[2] = 0.0
                        tl = t.norm()
                        if tl > FR * depth:            # Coulomb friction against the table
                            t *= FR * depth / tl
                        x[i] -= t
                v[i] = (x[i] - xp[i]) / DT

        @ti.kernel
        def substep(a: ti.f32):
            predict(a)
            solve_constraints()
            solve_tethers()
            solve_self()
            solve_contacts(a)

        @ti.kernel
        def substep_noself(a: ti.f32):
            predict(a)
            solve_constraints()
            solve_tethers()
            solve_contacts(a)

        pose_buf, BF, GF, MB = self.pose_buf, self.BOX_F, self.GRIP_F, self.max_boxes

        @ti.func
        def vec3(o):
            return ti.Vector([pose_buf[o], pose_buf[o + 1], pose_buf[o + 2]])

        @ti.func
        def mat3(o):
            return ti.Matrix([[pose_buf[o + 3 * r + c] for c in ti.static(range(3))]
                              for r in ti.static(range(3))])

        @ti.kernel
        def unpack_poses(n: ti.i32):
            """pose_buf -> colliders and gripper frames (layout: set_frame_poses)."""
            n_boxes[None] = n
            for b in range(n):
                o = b * BF
                b_c0[b] = vec3(o); b_R0[b] = mat3(o + 3)
                b_c1[b] = vec3(o + 12); b_R1[b] = mat3(o + 15)
                b_half[b] = vec3(o + 24)
            for k in range(2):
                o = MB * BF + k * GF
                g_c0[k] = vec3(o); g_R0[k] = mat3(o + 3)
                g_c1[k] = vec3(o + 12); g_R1[k] = mat3(o + 15)

        self._unpack = unpack_poses
        self._k = (build_hash, substep, substep_noself)

    # ---------------------------------------------------------------- API
    def frame(self):
        """Advance one 1/fps frame (substeps inside)."""
        build_hash, substep, substep_noself = self._k
        build_hash()                 # once per frame: points move < thickness per frame
        n, every = self.p.substeps, max(1, int(self.p.self_collision_every))
        for k in range(n):
            # progress through the frame (a kernel argument: no GPU sync); self-collision, the
            # costliest part, on every `every`-th substep and always on the last
            if k % every == every - 1 or k == n - 1:
                substep((k + 1) / n)
            else:
                substep_noself((k + 1) / n)

    def reset(self, X=None):
        X = self.X0 if X is None else np.asarray(X, np.float32)
        self.x.from_numpy(X); self.xp.from_numpy(X); self.target.from_numpy(X)
        self.v.fill(0)
        self.arm_fr.fill(0)                            # nothing held: the fingers don't grip
        self.teth_j.fill(-1)
        self._pin_np = np.full(self.N, FREE, np.int32)
        self._upload_pins()
        self._box_prev = self._grip_prev = None        # the robot poses start over too

    def _upload_pins(self):
        self.pin.from_numpy(self._pin_np)
        self.invm.from_numpy(np.where(self._pin_np == FREE, 1.0 / self.m, 0.0).astype(np.float32))

    # world-fixed pins (tests)
    def pin_world(self, idx, targets=None):
        idx = np.asarray(idx, int)
        self._pin_np[self._pin_np == WORLD] = FREE
        self._pin_np[idx] = WORLD
        self._upload_pins()
        if targets is not None:
            self.set_targets(idx, targets)

    def set_targets(self, idx, targets):
        t = self.target.to_numpy()
        t[np.asarray(idx, int)] = np.asarray(targets, np.float32)
        self.target.from_numpy(t)

    # gripper pins
    def pin_to_gripper(self, k: int, idx, local):
        """Points `idx` ride with gripper k at `local` (gripper-frame positions)."""
        idx = np.asarray(idx, int)
        pl = self.pin_local.to_numpy()
        pl[idx] = np.asarray(local, np.float32)
        self.pin_local.from_numpy(pl)
        self._pin_np[idx] = k
        self._upload_pins()
        self._set_tethers(k, idx)

    TETHER_SLACK = 0.02             # a tether is 2 % (+ 2 mm) longer than the fabric it spans

    def _set_tethers(self, k: int, idx):
        """Tether every point to its nearest pinched point of the same panel (a pinch of the top
        layer only must not lift the layer below through a shortcut)."""
        uv, panel = self.J["uv"], self._panel_np
        j = np.full(self.N, -1, np.int32)
        L = np.zeros(self.N, np.float32)
        if len(idx):
            for pnl in (0, 1):
                tgt = idx[(panel[idx] == pnl) | (panel[idx] == 2)]
                pts = np.where((panel == pnl) | (panel == 2))[0]
                if len(tgt) == 0:
                    continue
                D = np.linalg.norm(uv[pts, None, :] - uv[None, tgt, :], axis=2)
                a = np.argmin(D, axis=1)
                dist = D[np.arange(len(pts)), a]
                better = (j[pts] < 0) | (dist < L[pts])
                j[pts[better]] = tgt[a[better]]
                L[pts[better]] = dist[better]
            L = L * (1 + self.TETHER_SLACK) + 0.002
        tj, tl = self.teth_j.to_numpy(), self.teth_L.to_numpy()
        tj[:, k], tl[:, k] = j, L
        self.teth_j.from_numpy(tj); self.teth_L.from_numpy(tl)

    def release_gripper(self, k: int):
        if (self._pin_np == k).any():
            self._pin_np[self._pin_np == k] = FREE
            self._upload_pins()
            self._set_tethers(k, np.zeros(0, int))

    def set_box_sides(self, sides):
        """The arm (0 / 1, or -1) each collider box belongs to, in set_frame_poses order."""
        s = np.full(self.max_boxes, -1, np.int32)
        s[:len(sides)] = sides
        self.b_side.from_numpy(s)

    def set_arm_friction(self, k: int, mu: float):
        """Friction of arm k's boxes against the fabric (the others: gripper_friction)."""
        self.arm_fr[k] = mu

    def set_frame_poses(self, boxes, grippers):
        """Robot poses at the end of the coming frame: boxes (c (n,3), R (n,3,3), half (n,3)),
        grippers (c (2,3), R (2,3,3)). The previous poses become the start of the frame."""
        c, Rm, half = boxes
        n = len(c)
        if n > self.max_boxes:
            raise ValueError(f"{n} colliders, max {self.max_boxes}")
        if self._box_prev is None or len(self._box_prev[0]) != n:   # first frame: no motion yet
            self._box_prev = (c, Rm)
            self._grip_prev = grippers
        # per box: c0 R0 c1 R1 half (start / end of the frame), then per gripper: c0 R0 c1 R1
        flat = lambda a, k: np.asarray(a, np.float32).reshape(len(a), k)
        boxes_f = np.concatenate([flat(self._box_prev[0], 3), flat(self._box_prev[1], 9),
                                  flat(c, 3), flat(Rm, 9), flat(half, 3)], axis=1)
        grips_f = np.concatenate([flat(self._grip_prev[0], 3), flat(self._grip_prev[1], 9),
                                  flat(grippers[0], 3), flat(grippers[1], 9)], axis=1)
        buf = np.zeros(self.pose_buf.shape[0], np.float32)
        buf[:n * self.BOX_F] = boxes_f.ravel()
        buf[self.max_boxes * self.BOX_F:] = grips_f.ravel()
        self.pose_buf.from_numpy(buf)
        self._unpack(n)
        self._box_prev, self._grip_prev = (c, Rm), grippers

    def positions(self) -> np.ndarray:
        return self.x.to_numpy()


def _bend_pairs(faces: np.ndarray):
    """For each edge shared by two triangles, the two vertices opposite it."""
    opp = {}
    for a, b, c in faces:
        for u, w, o in ((a, b, c), (b, c, a), (c, a, b)):
            opp.setdefault((min(u, w), max(u, w)), []).append(int(o))
    return [(o[0], o[1]) for o in opp.values() if len(o) == 2]
