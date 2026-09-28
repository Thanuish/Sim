#!/bin/bash
# Opens the latest recorded episode (or the one given) in the MuJoCo viewer (macOS needs mjpython).
cd "$(dirname "$0")"
EP="${1:-$(.venv/bin/python scripts/episodes.py --latest 2>/dev/null)}"
[ -z "$EP" ] || [ ! -f "$EP" ] && EP="$(ls -t data/episode_*.hdf5 2>/dev/null | head -1)"
[ -z "$EP" ] && { echo "No episodes recorded yet."; exit 1; }
echo "Replaying $EP"
exec .venv/bin/mjpython scripts/replay.py "$EP"
