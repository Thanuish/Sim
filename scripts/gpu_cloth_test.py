"""GPU jeans test: the scripted fold without robots (pinned patches stand in for the grippers).

    python scripts/gpu_cloth_test.py                 # 1.2 cm, render to gpu_fold.png
    python scripts/gpu_cloth_test.py --spacing 0.01 --out fold_1cm

Reports the cost per 1/60 s frame (< 16.7 ms = faster than real time), cloth triangles passing
through each other (mesh_check), and saves a picture per phase (outside denim blue, inside pale:
any pale patch that isn't at an opening means the fabric passed through itself).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mesh_check as M  # noqa: E402
from vrteleop import cloth_config as CC  # noqa: E402
from vrteleop import gpu_cloth as G  # noqa: E402
from vrteleop import scene as S  # noqa: E402


def render(P, F, title, eye=(0.2, -1.1, 1.45), look=(0.5, 0.0, 0.76), size=(640, 420)):
    """Painter's-algorithm render: outward faces denim blue, inward faces pale."""
    from PIL import Image, ImageDraw
    W, H = size
    eye, look = np.array(eye), np.array(look)
    fwd = look - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    light = np.array([0.3, -0.5, 1.0]); light /= np.linalg.norm(light)

    def project(Q):
        d = Q - eye
        z = d @ fwd
        return np.c_[W / 2 + 820 * (d @ right) / z, H / 2 - 820 * (d @ up) / z], z

    img = Image.new("RGB", (W, H), (236, 234, 230)); dr = ImageDraw.Draw(img)
    tz = S.TABLE_Z
    q, _ = project(np.array([[0.05, -0.6, tz], [0.95, -0.6, tz], [0.95, 0.6, tz], [0.05, 0.6, tz]]))
    dr.polygon([tuple(p) for p in q], fill=(214, 212, 206))
    q, z = project(P)
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    outside = ((eye - P[F].mean(axis=1)) * n).sum(axis=1) > 0
    for f in np.argsort(-z[F].mean(axis=1)):
        nn = n[f] if outside[f] else -n[f]
        s = 0.35 + 0.65 * max(0.0, float(nn @ light))
        base = np.array([52, 82, 140]) if outside[f] else np.array([200, 206, 222])
        dr.polygon([tuple(q[k]) for k in F[f]], fill=tuple(int(c) for c in base * s))
    dr.text((8, 8), title, fill=(0, 0, 0))
    return img


class Fold:
    def __init__(self, cloth: G.GPUJeans):
        self.c = cloth
        self.phases = []
        self.snaps = []

    def patch(self, p, r=0.02):
        """All points (both layers) within r of p in the horizontal plane: a pinch."""
        X = self.c.positions()
        return np.where(np.linalg.norm(X[:, :2] - p[:2], axis=1) < r)[0]

    def run(self, name, seconds, moves=None):
        """moves: list of (idx, start (k,3), end (k,3)) moved along a smooth path."""
        c = self.c
        n = int(round(seconds * c.p.fps))
        t0 = time.perf_counter()
        for f in range(n):
            if moves:
                s = 0.5 - 0.5 * np.cos(np.pi * (f + 1) / n)
                for idx, a, b in moves:
                    c.set_targets(idx, a + (b - a) * s)
            c.frame()
        _ = c.x.to_numpy()[:1]                       # wait for the GPU
        ms = (time.perf_counter() - t0) / n * 1e3
        P = c.positions()
        cr = len(M.intersecting_pairs(P.astype(np.float64), c.faces))
        self.phases.append((name, ms, cr))
        self.snaps.append((name, P))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--garment", default=None)
    ap.add_argument("--spacing", type=float, default=0.012)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--thickness", type=float, default=1.0, help="self-collision distance / spacing")
    ap.add_argument("--arch", default="vulkan")
    ap.add_argument("--out", default="gpu_fold")
    a = ap.parse_args()
    G.init(a.arch)
    cfg = CC.load(a.garment)
    params = dataclasses.replace(cfg.gpu, spacing=a.spacing, substeps=a.substeps, thickness=a.thickness)
    cloth = G.GPUJeans(cfg.garment, params, (*S.JEANS_POS, 0.0), S.JEANS_YAW, S.TABLE_Z)
    print(f"garment '{cfg.name}': {cloth.N} points, {cloth.n_cons} constraints, {len(cloth.faces)} triangles,"
          f" spacing {a.spacing * 100:.1f} cm, thickness {cloth.thick * 1000:.1f} mm, {a.substeps} substeps")
    cloth.frame()                                    # compile the kernels
    fd = Fold(cloth)
    fd.run("settle", 1.0)
    # the fold of scripts/cloth_bench.py: far leg's waistband + hem corners, over the near leg
    X = cloth.positions()
    far = X[:, 0] > np.median(X[:, 0])
    ids = np.where(far)[0]
    w = ids[np.argmax(X[far, 1] * 5 + X[far, 0])]
    h = ids[np.argmax(-X[far, 1] * 5 + X[far, 0])]
    pw, ph = fd.patch(X[w] + [-0.02, -0.02, 0]), fd.patch(X[h] + [-0.02, 0.02, 0])
    cloth.pin_world(np.r_[pw, ph])
    aw, ah = X[pw], X[ph]
    up, over = np.array([0, 0, 0.16]), np.array([-0.40, 0, 0])
    fd.run("lift", 1.2, [(pw, aw, aw + up), (ph, ah, ah + up)])
    fd.run("carry", 2.2, [(pw, aw + up, aw + up + over), (ph, ah + up, ah + up + over)])
    # lay it onto the other leg (the pins are infinitely strong: pushed into the stack they'd
    # force the fabric through it, which real grippers can't)
    fd.run("lay down", 1.0, [(pw, aw + up + over, aw + over + [0, 0, 0.04]),
                             (ph, ah + up + over, ah + over + [0, 0, 0.04])])
    cloth.pin_world([])
    fd.run("release", 1.0)
    # hems up to the knees
    X = cloth.positions()
    hem = np.where(X[:, 1] < X[:, 1].min() + 0.03)[0]
    ph = fd.patch(np.r_[X[hem, 0].mean(), X[hem, 1].max() - 0.01, 0], r=0.025)
    cloth.pin_world(ph)
    a0 = X[ph]
    fd.run("hems lift", 1.2, [(ph, a0, a0 + [0, 0, 0.2])])
    tgt = a0 + [0, 0.02 - a0[:, 1].mean() + 0.0, 0.2]
    fd.run("hems carry", 2.2, [(ph, a0 + [0, 0, 0.2], tgt)])
    fd.run("hems down", 1.0, [(ph, tgt, tgt - [0, 0, 0.14])])
    cloth.pin_world([])
    fd.run("settle", 1.5)

    budget = 1e3 / cloth.p.fps
    print(f"\n  phase        ms per 1/60 s frame   crossings   (budget {budget:.1f} ms)")
    for name, ms, cr in fd.phases:
        print(f"  {name:12s} {ms:10.1f}             {cr:6d}")
    worst = max(ms for _, ms, _ in fd.phases)
    print(f"  real-time factor (worst phase): {budget / worst:.2f}")
    P = cloth.positions()
    print(f"  final footprint {np.ptp(P[:, 0]):.2f} x {np.ptp(P[:, 1]):.2f} m, "
          f"height {1e3 * (P[:, 2].max() - S.TABLE_Z):.0f} mm")
    from PIL import Image
    pick = [0, 2, 3, 4, 6, 8]
    tiles = [render(P_, cloth.faces, name) for i, (name, P_) in enumerate(fd.snaps) if i in pick]
    sheet = Image.new("RGB", (640 * 3, 420 * 2), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * 640, (i // 3) * 420))
    out = Path(a.out if a.out.endswith(".png") else a.out + ".png")
    sheet.save(out)
    print(f"  pictures -> {out}")


if __name__ == "__main__":
    main()
