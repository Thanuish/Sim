# MuJoCo VR Teleop — dual UR5e with Meta Quest

Teleoperate two simulated **UR5e arms with Robotiq 2F-85 grippers** in **MuJoCo** from a
**Meta Quest** headset, and record demonstrations as HDF5 episodes for imitation learning.

```
 Quest browser (WebXR, three.js)                 Mac / PC (Python)
 ┌───────────────────────────────┐   wss://   ┌──────────────────────────────────────┐
 │ renders MuJoCo bodies         │ ◄───────── │ body poses @ 90 Hz                   │
 │ controller + head poses @72Hz │ ─────────► │ clutch mapping → DLS IK (100 Hz)     │
 │ buttons: record / reset …     │            │ MuJoCo physics (500 Hz, real time)   │
 └───────────────────────────────┘            │ HDF5 episode recorder (30 Hz)        │
                                              └──────────────────────────────────────┘
```

All physics, IK and recording run in Python; the headset is a thin client, so what you
record is exactly what MuJoCo simulated. Nothing has to be installed on the Quest.

## Tasks

| `--task` | What you do |
|---|---|
| `jeans` (default) | Fold a pair of denim jeans lying on the table |
| `blocks` | Pick three cubes and a cylinder and drop them into the bin (the original task) |

### Jeans folding

The jeans are a full-size pair (1.0 m long, 0.45 m flat hip width, 0.65 kg) built like real
jeans: a **front and a back panel sewn together** along the outer leg seams and the inseams
(through the crotch), **open at the waist and at both hems**, so the legs are tubes. The front
panel shows the fly, front pockets, coin pocket and rivets; the back panel the yoke, back pockets
and leather patch; the inside is the pale reverse side of the denim. Physics:

- threads along and across the legs barely stretch (about 1 %, a few % at most while a leg hangs
  from the grippers); the fabric shears easily on the
  bias, so it drapes and folds; denim bending stiffness; air drag on falling fabric;
- friction against the laminate table (μ≈0.4) and the silicone finger pads (μ≈1.0);
- the fabric never passes through itself: not the two panels, not a leg folded over the other,
  not a crumpled heap (a contact sphere at every vertex touches the continuous surface of both
  panels, and the surface also collides with itself triangle against triangle). The whole
  gripper (fingers and palm, not only the pads) touches that surface, so a finger can't slip
  through the fabric and snag it. Fabric sliding on fabric is frictionless (`self_condim` in
  `GarmentSpec` turns friction on at ~20 % more cost).

The simulation resolution is 5.5 cm (`--cloth-spacing`, about 280 vertices) with a 4 ms physics
step. When the CPU can't keep up the server runs in slight slow motion; recorded data stays
consistent. Closing other apps and turning off the camera screens (`--stream-cams ""`) helps;
`--cloth-spacing 0.06` is faster but coarser. `--cloth-fast` uses the older collision model
(~40 % less CPU), in which folded or crumpled fabric can cut through itself and the pale inside
shows.

**GPU cloth engine** (the default of `start.bat`; `start.bat --cloth-engine mujoco` for the
MuJoCo cloth above, which the server also falls back to if the GPU engine can't start). The
jeans are simulated on the GPU (`vrteleop/gpu_cloth.py`, XPBD in Taichi on any Vulkan GPU, AMD
included) at 1 cm resolution: ~9000 points instead of ~300, so folds are round and creases
sharp. MuJoCo then only
simulates the robots; every 1/60 s the arms' and grippers' collision shapes (the curved finger links
as several boxes each) are handed to the cloth, which collides with them, with the table and with
itself (with friction, so folds stay put). The pinch works as below and holds every point of every
layer between the pads. While a gripper pinches, every point is tethered to its nearest pinched
point (long-range attachments, Kim et al. 2012), so a lifted leg hangs at its true length
instead of stretching like rubber and snapping back; and only an arm that pinches has friction
on the fabric (open fingers don't squeeze it, so it slides off them). Settings: `[gpu]` in
`config/cloth.toml`. Headless check:
`python scripts/gpu_cloth_test.py`; with the teleop controls of both arms, including replays of
recorded VR sessions: `python scripts/teleop_trials.py --cloth-engine gpu --human EPISODE.hdf5`.
Not yet with this engine: `--randomize` / `--cloth-init crumpled` start flat, and a pinch doesn't
slip. Episodes are larger (~1 MB per second).

**The folding game** (jeans task). The panels show three steps, checked on the footprint of
the jeans on the table once the fabric has settled and nothing is held: 1. lay one leg over the
other (footprint ≤ 62 % of the flat jeans), 2. fold them in half, hems up to the waist (≤ 35 %),
3. fold once more (≤ 22 %). The clock starts when you first grip an arm and stops at the last
step; the best time is kept while the server runs. X (reset scene) starts a new game.
`vrteleop/fold_game.py` has the steps and thresholds.

**How grasping works.** Lower the open gripper until the fingertips are just above the fabric,
then close it. The 2F-85 fingertips swing down about 18 mm as they close, and that motion
pinches the fabric. The jaws only catch fabric if they close with cloth between the fingertips,
so a closed gripper pushed onto the jeans does not grab them. The fabric slips out if you pull
harder than a pinch can hold (`--slip-force`, default 30 N), and it is released as soon as you
start opening the gripper (once the pads are a few millimetres further apart than when they
pinched, so a short flick of the stick is enough). Like real jaws, a pinch holds every layer
between the pads: both panels at an edge, or all layers of a folded stack. This is a modelled
pinch: the 5 cm simulation mesh cannot form the millimetre-scale
fold that real jaws squeeze, so friction alone would never hold it. Everything else is plain
physics. The wrist panel shows ✋ while a gripper is holding fabric.

**A typical fold:** pinch the far leg at the waistband (left arm) and at the hem (right arm),
lift it and lay it over the near leg. Then pinch both hems with the right arm and fold them up
to the knees or the waist.

Options: `--cloth-init crumpled` drops the jeans into a random heap on reset.
`--cloth-spacing 0.045` gives a finer cloth but needs a fast CPU. `--randomize` also jitters the
jeans' heading. The laptop page shows how much the footprint has shrunk (the fold progress).

**Cloth settings file.** Every garment and cloth-simulation value lives in `config/cloth.toml`,
each with a comment on what it does, its unit and sensible values: the garment's size, mass,
stretch, bending, shear, friction, contact settings, the physics timestep and solver budget, and
the grip (slip force, release gap, pad friction). Edit a value and restart the server. The file
defines several garments (`jeans`, `shorts`, `stretch_jeans`); add your own as a new
`[garments.NAME]` table, usually starting from an existing one with `inherits = "jeans"`, and
start it with `start.bat --garment NAME` (`--cloth-config FILE` uses another file). Misspelled
keys and impossible sizes are reported when the server starts. Every episode stores the exact
settings it was recorded with, so replays stay correct after you edit the file. All garments use
the jeans pattern (two panels, two legs); the file sets their size and fabric. Check a new garment
without the headset: `python scripts/cloth_bench.py --garment NAME --test all`.

Episodes additionally contain `observations/cloth_verts` (T, N, 3), `teleop/cloth_grasp`
(T, 2, 12: the pinched vertices of every layer between the pads, -1 = none), the flat pattern (`cloth/faces`, `cloth/rest_uv`) and the attributes `task`,
`final_coverage` (footprint / flat footprint: 1 = spread out, about 0.25 = folded in quarters)
and `final_height`. `replay.py --info` prints the coverage. Cloth is chaotic, so
`--mode actions` re-simulation diverges from the recording; use the default `states` replay
for exact playback.

Headset-free check of the whole pipeline: `python scripts/fake_client.py --fold` (with the
server running with `--http`).

Cloth physics benchmark: `python scripts/cloth_bench.py --test all` runs the same fold headless
in simulation time (results don't depend on the machine) and reports thread stretch,
penetration, cloth triangles passing through each other, jitter at rest, slips, whether released
fabric sticks to the pads, the final fold coverage and the step cost per phase; `--test release`
and `--test poke` repeat pinch/open and push/sweep trials. `--out sheet` also saves side-view snapshots. Use it to
check changes to the cloth or the grasp model.

### Realism

- A room with an oak plank floor, painted walls, a ceiling light, and a laminate lab table on
  an aluminium frame. The textures are procedural and generated into `assets/generated/` on
  start-up.
- MuJoCo renders (camera feeds and recorded images) use soft shadows and 8× anti-aliasing.
- The headset view uses physically based materials, image-based reflections, soft shadows and
  filmic tone mapping. The jeans get twill texture, topstitching, pockets, rivets, whiskers, a
  back side with back pockets and a leather patch, and a fabric sheen. The cloth is smoothed
  on the client with Loop subdivision. Add `?lite` to the URL on older headsets to turn off
  shadows and reflections.
- Physics has priority: camera screens render only while the simulation keeps real time, so
  the feeds may run at a few fps during heavy cloth manipulation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_assets.py        # only needed if ./assets is missing (UR5e + 2F-85 from MuJoCo Menagerie)
```

## Windows

1. Install [Python 3.12](https://www.python.org/downloads/) (tick **Add python.exe to PATH**) and
   [Android platform-tools](https://developer.android.com/tools/releases/platform-tools) (add the
   folder to PATH so `adb` works).
2. Get the project (`git clone ...` or copy the folder, without `.venv`).
3. Double-click **setup_windows.bat** once.
4. Plug in the Quest with USB-C and double-click **start.bat** (it runs `adb reverse` for you).
   Open `http://localhost:8080` in the Quest Browser.
   For Wi-Fi instead: `start.bat --https` and open `https://<PC-IP>:8443` on the Quest.
5. **replay.bat** opens the newest recording in the MuJoCo viewer.

## Run with Docker (any PC: Windows, Linux or Mac)

Physics runs on the CPU, so a faster CPU means a smoother simulation. Install
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine
(Linux), copy this folder to the PC, and in it run:

```bash
docker compose build          # once (and after changing the code)
docker compose up             # starts the server on http://localhost:8080
```

**Quest over USB-C (recommended):** install Android platform-tools on the PC, plug in the Quest
and run `adb reverse tcp:8080 tcp:8080`, then open `http://localhost:8080` in the Quest Browser.

**Quest over Wi-Fi:** WebXR needs HTTPS for a network address, and the container can't see the
PC's IP, so pass it in (find it with `ipconfig` on Windows, `ip a` on Linux):

```bash
TELEOP_PUBLIC_HOST=192.168.1.50 TELEOP_ARGS="" docker compose up      # Linux/Mac shell
```
```powershell
$env:TELEOP_PUBLIC_HOST="192.168.1.50"; $env:TELEOP_ARGS=""; docker compose up   # Windows PowerShell
```
Then open `https://192.168.1.50:8443` on the Quest and accept the certificate warning once.

**NVIDIA GPU:** `docker compose up teleop-gpu` renders the camera screens and videos on the GPU
(needs the NVIDIA driver + NVIDIA Container Toolkit; on Windows, Docker Desktop with WSL2).
Without a GPU the CPU version renders them in software, which is slower; physics has priority,
so camera screens just update less often.

Recorded episodes land in the PC's `data/` folder (mounted into the container). Extra server
options go in `TELEOP_ARGS`, e.g. `TELEOP_ARGS="--http --task blocks"`.

## Run

```bash
python -m vrteleop.server
```

It prints something like `https://192.168.1.23:8443`.

1. Put the Quest on the **same Wi-Fi** as your computer (on macOS, allow incoming connections for Python if asked).
2. Open that URL in the **Quest Browser**. You get a certificate warning because the certificate is self-signed.
   Choose **Advanced → Proceed**. You only have to do this once.
3. Press **Enter VR**. Stand up; the robots are mounted on a table right in front of you.
   Press **Y** to recenter the scene on where you are standing and facing.

**Connecting over USB instead of Wi-Fi** (lower latency, and no certificate needed):

```bash
adb reverse tcp:8080 tcp:8080
python -m vrteleop.server --http
# Quest Browser → http://localhost:8080
```

**Testing without the headset:** open `https://localhost:8443` in a desktop browser to watch the scene
(orbit with the mouse). To smoke-test the full loop, run `python scripts/fake_client.py --url wss://localhost:8443/ws`.
It drives both arms in circles and records one episode.

## Controls

| Input | Action |
|---|---|
| **Grip** (hold) | Engage that arm (clutch). While held, your hand motion moves that gripper. Release it to reposition your hand. |
| **Thumbstick ↓ / ↑** | Close / open that arm's gripper. It stays where you leave it. |
| **Thumbstick click** | Show or hide the live camera screens |
| **A** (right) | Start recording / stop and save the episode as a *success* |
| **B** (right) | Discard the current episode and reset |
| **X** (left) | Reset the scene (objects get re-randomized) |
| **Y** (left) | Recenter: put your eyes back at the robot's head |
| **Left trigger + left stick** | Move your viewpoint forward/back and sideways (remembered in the browser) |
| **Right trigger + right stick ↑/↓** | Move your viewpoint up/down |
| **Left trigger + Y** | Back to the default viewpoint |
| Desktop page | Record / Save as fail / Discard / Reset buttons. `Space` = record, `R` = reset |

**First-person view:** when you enter VR, the scene is aligned automatically so that your eyes sit
at the robot's "head" (`HEAD_POS` in `scene.py`): centred between the two arm bases, slightly behind
and above them, looking at the table. The arms then reach out in front of you like your own arms.
If you walk around or the view drifts, press **Y**.

**Watching on the laptop:** while someone teleoperates in VR, open `http://localhost:8080` (or the
https address) in a browser on the laptop. The view automatically follows the headset wearer's head.
Press `V` to switch between following the headset and free mouse orbit, `H` to hide the panels and `F` for
fullscreen. The live camera feeds appear as thumbnails, and the Record / Reset buttons work from the laptop too.

**Camera screens:** three live MuJoCo camera feeds (left wrist, head, right wrist) float beyond
the far edge of the table. The desktop page shows the same feeds as thumbnails. Pick different
cameras with `--stream-cams`, for example `--stream-cams overhead,front`. Use `--cam-stream-hz` to set
the frame rate (default 15) and `--img-w/--img-h` to set the resolution. If rendering is slow, the stream
rate drops automatically so that the physics never falls behind.

The left wrist shows a small panel with the recording state, the gripper openings and which arms are engaged.
When an arm is engaged, its target pose is drawn as a small axis marker.

**How the mapping works:** the clutch mapping is relative. When you squeeze the grip, the controller's
pose at that moment is paired with the gripper's current pose. From then on, the controller's
translation and rotation relative to that moment are applied to the gripper, in the world frame.
Because of this you never need exact calibration between your body and the robot, and you can
"ratchet" through large motions. Use `--scale 1.5` to amplify your hand motion.

## Recorded data

### Where episodes go

Every server run is a session folder; every saved demo gets a summary and a preview video:

```
data/
  jeans/                       one folder per task
    index.csv                  one row per saved episode, all sessions (open it in Excel/Numbers)
    2026-09-28_14-05-12/       one folder per server run
      session.json             operator, machine, MuJoCo version, all server settings
      episode_0000.hdf5        the demonstration (layout below)
      episode_0000.json        success, duration, frames, fold coverage
      episode_0000.mp4         first-person video of what the operator saw in the headset
```

Options: `--session NAME` (a fixed folder name; reuse it to add to an existing session),
`--operator NAME` (defaults to your login name), `--no-preview` (skip the MP4 videos; render one later with
`python scripts/replay.py EPISODE.hdf5 --video out.mp4 --camera operator`), `--data-dir PATH`.
`python scripts/episodes.py` summarises everything recorded (`--list`, `--failed`, `--latest`), and
`./replay.command` opens the newest episode in the viewer.

### What is inside each episode

Each `episode_XXXX.hdf5` contains:

| Key | Shape | Content |
|---|---|---|
| `observations/qpos` | (T, 14) | `[L arm 6, L gripper 0..1, R arm 6, R gripper 0..1]` |
| `observations/qvel` | (T, 14) | Joint velocities |
| `observations/ee_pos`, `ee_quat` | (T, 2, 3/4) | Measured gripper pinch-point pose (wxyz) |
| `observations/object_pose` | (T, 4, 7) | Cube and cylinder poses |
| `observations/images/<cam>` | (T, H, W, 3) | Only with `--cameras` |
| `observations/full_qpos/qvel` | (T, nq/nv) | Full MuJoCo state, for exact replay |
| `action` | (T, 14) | Joint-space command `[L q 6, L grip, R q 6, R grip]` |
| `action_ee/pos, quat, gripper` | | End-effector targets the operator commanded |
| `teleop/engaged, head_pose, controller_pose` | | Raw VR signals |

Root attributes: `fps`, `timestep`, `success`, `joint_names`, `object_names`, `camera_names`, `model_xml`.
The layout follows ALOHA/ACT conventions, so converting to LeRobot is straightforward.

The available cameras are `head_cam` (the first-person view), `overhead`, `front`, `left_gripper_wrist_cam` and `right_gripper_wrist_cam`:

```bash
python -m vrteleop.server --cameras                                    # all cameras, 320x240
python -m vrteleop.server --cameras --camera-names front,left_gripper_wrist_cam,right_gripper_wrist_cam --img-w 640 --img-h 480
```

Rendering images costs a few milliseconds per camera per frame. If the real-time factor (`rtf` on the
desktop page) falls below 1, record fewer or smaller cameras.

### Replay and inspect

```bash
python scripts/replay.py data/episode_0000.hdf5 --info                   # list datasets
python scripts/replay.py data/episode_0000.hdf5 --video ep0.mp4 --camera overhead
mjpython scripts/replay.py data/episode_0000.hdf5                        # interactive viewer (macOS needs mjpython)
python scripts/replay.py data/episode_0000.hdf5 --mode actions --video check.mp4   # re-simulate from actions
```

## Server options

```
--http               plain HTTP on :8080 (use with adb reverse)
--port N             port override
--scale 1.0          hand → robot translation scale
--randomize 0.05     object xy jitter on reset [m]
--fps 30             recording rate (sim time)
--control-hz 100     IK / control rate
--stream-hz 90       pose streaming rate to the headset
--stream-cams ...    live camera screens in VR (default: left wrist, head, right wrist; '' = off)
--cam-stream-hz 15   camera screen frame rate
--data-dir data      output folder
--garment NAME       jeans task: garment from the cloth settings (default: the file's `default`)
--cloth-config FILE  jeans task: cloth settings file (default config/cloth.toml)
--cloth-engine NAME  jeans task: cloth simulation, gpu (default) or mujoco
--cloth-spacing M    jeans task: override the garment's simulation resolution
--cloth-fast         jeans task: cheaper cloth collisions (folds can cut through themselves)
--slip-force N       jeans task: override the pinch slip force
--cloth-init crumpled  jeans task: start from a random heap
```

## Control from Claude (MCP) and the control API

The running server has a small HTTP control API, reachable from this PC only:
`GET /api/status`, `POST /api/command {"cmd": "reset" | "record_toggle" | "save_fail" | "discard" |
"cams_toggle"}`, `GET /api/cloth`, and `POST /api/reload {"garment": "shorts", "overrides": {...}}`,
which rebuilds the scene with another garment (re-reading `config/cloth.toml`) without restarting;
the headset picks up the new scene by itself.

`vrteleop/mcp_server.py` is an MCP server on top of it, so Claude (or any MCP client) can run and
tune the simulator: `sim_status`, `sim_command`, `load_garment`, `list_garments`,
`cloth_settings`, `set_cloth_value` (edits `config/cloth.toml` keeping its comments, validates
the file before saving, `apply=true` reloads the running scene), `run_cloth_test` (the physics
benchmark), `list_episodes`, `episode_info`. It is a separate process and only talks to the
control API, so the simulator doesn't depend on it. Claude Code picks it up from `.mcp.json` in
the repo (approve the `teleop-sim` server once); on macOS/Linux change its `command` to
`.venv/bin/python`. Settings and tests work without the simulator running; `sim_*` and
`load_garment` need `start.bat` running.

## Project layout

```
vrteleop/
  server.py     orchestrates the parts: real-time sim loop, recording, headset streaming,
                control API
  arms.py       the two teleoperated arms: clutch mapping + IK -> joint / gripper commands
  cloth_engine.py  ClothEngine interface (reset, update, step, verts, held, metrics) with the
                MuJoCo and the GPU engines; the server only talks to this interface
  fold_game.py  the folding game: steps, thresholds, clock
  scene.py      MjSpec scene: room, table, 2x UR5e + 2F-85, cameras, home poses, tasks
  cloth.py      jeans pattern, flex cloth, pinch grasp model, fold metrics, denim textures
  cloth_config.py  loads garments and cloth settings from config/cloth.toml
  gpu_cloth.py  XPBD cloth on the GPU (Taichi/Vulkan), ~1.2 cm resolution
  mcp_server.py MCP server for Claude / MCP clients (talks to the control API)
  textures.py   procedural floor / table textures
  ik.py         damped-least-squares differential IK (runs on a kinematic shadow MjData)
  teleop.py     clutch-based controller → EE target mapping, workspace limits
  recorder.py   HDF5 episode writer
  webscene.py   exports MuJoCo visual geometry (meshes deduplicated) to the web client
  certs.py      self-signed certificate generation
  web/          WebXR client (three.js r170 vendored, works offline)
scripts/
  replay.py     inspect / replay / render episodes
  fake_client.py  headset-free end-to-end test
  cloth_bench.py  headless cloth physics benchmark (stretch, penetration, grasp release, cost)
  mesh_check.py   triangle-mesh self-intersection test (used by cloth_bench.py)
  gpu_cloth_test.py  scripted fold of the GPU cloth (speed, self-crossings, pictures)
  teleop_trials.py   pinch / release / poke trials through the teleop controls, any cloth
                     engine; --human replays recorded VR sessions into it
config/
  cloth.toml    garments (size, fabric) and cloth simulation / grasp settings
assets/         MuJoCo Menagerie models (UR5e, Robotiq 2F-85; see their LICENSE files)
```

## Customizing

- **Scene and task:** tasks live in `TASKS` / `build_spec()` in `vrteleop/scene.py` (blocks: `OBJECTS`, `BIN_POS`;
  jeans: `JEANS_POS`; the garment's size, mass and stiffness in `config/cloth.toml`). The web client
  picks up any geometry change automatically, because it receives the compiled model.
- **Robot mounting:** `ARM_Y`, `ARM_X`, `TABLE_Z` and `HOME_Q` in `scene.py`.
- **Controller feel:** `DiffIK` gains and `max_joint_vel` in `ik.py`. The servo damping of the arms is
  reduced in `build_spec()` to kv = kp/10 so they respond faster.
- **Where your eyes are:** `HEAD_POS` in `scene.py` and `OPERATOR_EYE_MJ` in `web/main.js`. Keep the two in sync.

## Notes

- Tested with MuJoCo 3.3+ (developed on 3.14), Python 3.10+. Any modern Quest (2/3/3S/Pro) with the Quest Browser should work.
- The controller 3D models are fetched from the WebXR input-profiles CDN. Offline, the controllers
  are invisible, but tracking still works.
- The server disengages both arms when controller data stops for more than 0.35 s, for example when the headset sleeps.
