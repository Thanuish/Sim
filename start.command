#!/bin/bash
# Starts the teleop server. Double-click in Finder, or: ./start.command [--cameras] [--http]
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run setup_mac.command first."; read -r -p "Press Enter to close"; exit 1
fi
exec .venv/bin/python -m vrteleop.server "$@"
