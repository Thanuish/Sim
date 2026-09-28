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

- threads along and across the legs barely stretch (under 1 %); the fabric shears easily on the
  bias, so it drapes and folds; denim bending stiffness; air drag on falling fabric;
- friction against the laminate table (μ≈0.4) and the silicone finger pads (μ≈1.0);
- the two panels can't pass through each other or through folded layers (contact spheres on
  the front panel against the continuous surface of the back panel, plus sphere-sphere contact);
  fabric sliding on fabric is frictionless (`self_condim` in `GarmentSpec` turns friction on at
  ~20 % more cost).

The simulation resolution is 5.5 cm (`--cloth-spacing`, about 280 vertices). On the development
Mac it runs at about real time, and 0.7-0.9x during the heaviest two-arm moves (the server then
runs in slight slow motion; recorded data stays consistent). Closing other apps and turning off
the camera screens (`--stream-cams ""`) helps; `--cloth-spacing 0.06` is faster but coarser.

**How grasping works.** Lower the open gripper until the fingertips are just above the fabric,
then close it. The 2F-85 fingertips swing down about 18 mm as they close, and that motion
pinches the fabric. The jaws only catch fabric if they close with cloth between the fingertips,
so a closed gripper pushed onto the jeans does not grab them. The fabric slips out if you pull
harder than a pinch can hold (`--slip-force`, default 30 N), and it is released when you open
the gripper. This is a modelled pinch: the 5 cm simulation mesh cannot form the millimetre-scale
fold that real jaws squeeze, so friction alone would never hold it. Everything else is plain
physics. The wrist panel shows ✋ while a gripper is holding fabric.

**A typical fold:** pinch the far leg at the waistband (left arm) and at the hem (right arm),
lift it and lay it over the near leg. Then pinch both hems with the right arm and fold them up
to the knees or the waist.

Options: `--cloth-init crumpled` drops the jeans into a random heap on reset.
`--cloth-spacing 0.04` gives a finer cloth but needs a fast CPU. `--randomize` also jitters the
jeans' heading. The laptop page shows how much the footprint has shrunk (the fold progress).

Episodes additionally contain `observations/cloth_verts` (T, N, 3), `teleop/cloth_grasp`
(T, 2, 3), the flat pattern (`cloth/faces`, `cloth/rest_uv`) and the attributes `task`,
`final_coverage` (footprint / flat footprint: 1 = spread out, about 0.25 = folded in quarters)
and `final_height`. `replay.py --info` prints the coverage. Cloth is chaotic, so
`--mode actions` re-simulation diverges from the recording; use the default `states` replay
for exact playback.

Headset-free check of the whole pipeline: `python scripts/fake_client.py --fold` (with the
server running with `--http`).

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
```

## Project layout

```
vrteleop/
  scene.py      MjSpec scene: room, table, 2x UR5e + 2F-85, cameras, home poses, tasks
  cloth.py      jeans pattern, flex cloth, pinch grasp model, fold metrics, denim textures
  textures.py   procedural floor / table textures
  ik.py         damped-least-squares differential IK (runs on a kinematic shadow MjData)
  teleop.py     clutch-based controller → EE target mapping, workspace limits
  recorder.py   HDF5 episode writer
  webscene.py   exports MuJoCo visual geometry (meshes deduplicated) to the web client
  server.py     aiohttp HTTPS + WebSocket server and the real-time sim loop
  certs.py      self-signed certificate generation
  web/          WebXR client (three.js r170 vendored, works offline)
scripts/
  replay.py     inspect / replay / render episodes
  fake_client.py  headset-free end-to-end test
assets/         MuJoCo Menagerie models (UR5e, Robotiq 2F-85; see their LICENSE files)
```

## Customizing

- **Scene and task:** tasks live in `TASKS` / `build_spec()` in `vrteleop/scene.py` (blocks: `OBJECTS`, `BIN_POS`;
  jeans: `JEANS_POS`, and the garment's size, mass and stiffness in `GarmentSpec` in `vrteleop/cloth.py`). The web client
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
