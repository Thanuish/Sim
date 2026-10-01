"""MCP server for the teleop simulator: lets Claude (or any MCP client) run and tune it.

    python -m vrteleop.mcp_server            # stdio transport (what MCP clients start)

Registered for Claude Code in the repo's .mcp.json. It is a separate process that talks to the
running simulator over its control API (http://localhost:8080/api/..., this PC only; override
with SIM_URL), so the simulator never depends on MCP. Tools that don't need a running simulator
(cloth settings, tests, recordings) work on the files directly.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP

from . import cloth_config as CC

ROOT = Path(__file__).resolve().parents[1]
SIM_URL = os.environ.get("SIM_URL", "http://localhost:8080").rstrip("/")
PYTHON = sys.executable

mcp = FastMCP("teleop-sim", instructions=(
    "Controls the MuJoCo dual-UR5e VR teleoperation simulator (jeans folding). sim_* tools need "
    "the simulator running (start.bat); cloth settings live in config/cloth.toml and take effect "
    "with load_garment (or apply=True on set_cloth_value). run_cloth_test measures cloth physics "
    "headless and takes a few minutes."))


# ------------------------------------------------------------------ simulator (control API)
def _api(method: str, path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    try:
        r = httpx.request(method, f"{SIM_URL}{path}", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise RuntimeError(f"the simulator isn't reachable at {SIM_URL} ({e.__class__.__name__}); "
                           f"start it with start.bat") from None
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(data.get("error", f"HTTP {r.status_code}"))
    return data


@mcp.tool()
def sim_status() -> dict:
    """Current simulator state: garment, real-time factor (rtf, 1 = real time), recording,
    grippers (engaged, closure, holding cloth), fold progress (coverage: 1 flat .. ~0.3 folded),
    headset input, last message."""
    s = _api("GET", "/api/status")
    return {k: s.get(k) for k in ("garment", "cloth_engine", "task", "rtf", "sim_time", "recording", "rec_time",
                                  "next_episode", "last_saved", "engaged", "gripper", "holding",
                                  "cloth", "game", "input_fresh", "input_hz", "cams_on", "message")}


@mcp.tool()
def sim_command(command: Literal["reset", "record_toggle", "save_fail", "discard", "cams_toggle"]) -> str:
    """Send a command, like the headset buttons: reset the scene (discards a recording),
    record_toggle (start / stop+save an episode), save_fail (stop and save as failed),
    discard (drop the recording and reset), cams_toggle (camera screens on/off)."""
    return _api("POST", "/api/command", {"cmd": command})["message"]


@mcp.tool()
def load_garment(garment: str | None = None, overrides: dict[str, Any] | None = None) -> str:
    """Rebuild the running scene with a garment from config/cloth.toml (re-read, so file edits
    apply), optionally with one-off overrides of garment settings, e.g. {"mass": 0.8}.
    Not allowed while recording. The headset picks up the new scene by itself."""
    return _api("POST", "/api/reload", {"garment": garment, "overrides": overrides or {}},
                timeout=60.0)["message"]


# ------------------------------------------------------------------ cloth settings (file)
@mcp.tool()
def list_garments() -> list[str]:
    """Garments defined in config/cloth.toml."""
    return CC.names()


@mcp.tool()
def cloth_settings(garment: str | None = None) -> dict:
    """All resolved settings of a garment (after `inherits`), plus the simulation and grasp
    settings. Default: the file's default garment."""
    cfg = CC.load(garment)
    return json.loads(cfg.to_json())


def _format(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    return json.dumps(str(value))


def set_toml_value(text: str, section: str, key: str, value) -> str:
    """Set `key = value` in [section] of a TOML text, keeping comments and layout."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    head = re.compile(r"^\s*\[\s*" + re.escape(section) + r"\s*\]\s*(#.*)?$")
    start = next((i for i, ln in enumerate(lines) if head.match(ln)), None)
    if start is None:                                   # new section at the end
        return text.rstrip(nl) + f"{nl}{nl}[{section}]{nl}{key} = {_format(value)}{nl}"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    item = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*)([^#]*?)(\s*#.*)?$")
    for i in range(start + 1, end):
        m = item.match(lines[i])
        if m:
            old = m.group(2)
            new = _format(value)
            comment = m.group(3) or ""
            if comment:                                  # keep the comment column where possible
                pad = max(1, len(old) + len(comment) - len(comment.lstrip()) - len(new))
                comment = " " * pad + comment.lstrip()
            lines[i] = m.group(1) + new + comment
            return nl.join(lines)
    last = max(i for i in range(start, end) if lines[i].strip()) if end > start else start
    lines.insert(last + 1, f"{key} = {_format(value)}")
    return nl.join(lines)


@mcp.tool()
def set_cloth_value(key: str, value: Any, garment: str | None = None, apply: bool = False) -> str:
    """Change one setting in config/cloth.toml, keeping all comments. Garment settings (mass,
    bend_young, edge_solref, spacing, ...) go into [garments.<garment>] (default: the file's
    default garment); timestep / solver_iterations go into [simulation], slip_force /
    release_gap / pad_friction into [grasp]. The whole file is validated before it is saved.
    apply=True also reloads the running simulator with that garment."""
    path = CC.DEFAULT_PATH
    text = path.read_text(encoding="utf-8")
    section = next((s for s, keys in CC.SECTIONS.items() if key in keys), None)
    target = garment or _default_garment(text)
    if section is None:
        section = f"garments.{target}"
    new_text = set_toml_value(text, section, key, value)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8",
                                     newline="") as f:
        f.write(new_text)
        tmp = Path(f.name)
    try:
        for name in CC.names(tmp):                      # every garment must still load
            CC.load(name, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    path.write_text(new_text, encoding="utf-8", newline="")
    msg = f"set {key} = {_format(value)} in [{section}]"
    if apply:
        msg += "; " + load_garment(target)
    return msg


def _default_garment(text: str) -> str:
    m = re.search(r'^\s*default\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise ValueError("config/cloth.toml has no default garment; pass garment=")
    return m.group(1)


# ------------------------------------------------------------------ tests and recordings
@mcp.tool()
def run_cloth_test(test: Literal["fold", "release", "poke", "all"] = "fold",
                   garment: str | None = None, extra_args: list[str] | None = None) -> str:
    """Run the headless cloth physics benchmark (scripts/cloth_bench.py) and return its report:
    thread stretch, penetration, self-crossings, jitter, slips, sticking, fold coverage, cost per
    simulated second. Takes ~2-5 min for 'fold', longer for 'all'. extra_args go to the server
    options, e.g. ["--cloth-init", "crumpled"] or ["--cloth-fast"]."""
    cmd = [PYTHON, "-W", "ignore", str(ROOT / "scripts" / "cloth_bench.py"), "--test", test, "--no-snapshots"]
    if garment:
        cmd += ["--garment", garment]
    cmd += list(extra_args or [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    out = "\n".join(ln for ln in r.stdout.splitlines() if not ln.startswith(("[scene]", "[data]")))
    if r.returncode != 0:
        out += "\n" + r.stderr[-3000:]
    return out.strip()


@mcp.tool()
def list_episodes(limit: int = 10) -> list[dict]:
    """Most recent recorded episodes (newest first) with their summary."""
    data = ROOT / "data"
    eps = sorted(data.rglob("episode_*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in eps:
        meta = p.with_suffix(".json")
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        info.pop("path", None)
        out.append({**info, "path": str(p.relative_to(ROOT)), "size_mb": round(p.stat().st_size / 1e6, 1),
                    "has_video": p.with_suffix(".mp4").exists()})
    return out


@mcp.tool()
def episode_info(path: str) -> dict:
    """Attributes and dataset shapes of a recorded episode (path relative to the repo)."""
    import h5py
    p = (ROOT / path).resolve()
    if ROOT not in p.parents:
        raise ValueError("path must be inside the project")
    with h5py.File(p, "r") as h:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()
                 if k not in ("model_xml", "cloth_config")}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
        shapes = {}
        h.visititems(lambda n, o: shapes.__setitem__(n, list(o.shape)) if hasattr(o, "shape") else None)
    return {"attributes": attrs, "datasets": shapes}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
