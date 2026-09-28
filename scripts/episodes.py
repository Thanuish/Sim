"""Browse recorded episodes (reads the index.csv files written by the server).

    python scripts/episodes.py                  # summary per task and session
    python scripts/episodes.py --list           # every episode
    python scripts/episodes.py --list --failed  # only episodes saved as failures
    python scripts/episodes.py --latest         # path of the newest episode
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(data_dir: Path):
    rows = []
    for index in sorted(data_dir.glob("*/index.csv")):
        with index.open() as f:
            for r in csv.DictReader(f):
                r["_abs"] = str(data_dir / r["path"])
                r["_exists"] = Path(r["_abs"]).exists()
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--failed", action="store_true")
    p.add_argument("--latest", action="store_true")
    a = p.parse_args()
    rows = [r for r in load(Path(a.data_dir)) if r["_exists"]]
    if a.failed:
        rows = [r for r in rows if r["success"] != "True"]
    if not rows:
        print("no episodes recorded yet")
        return
    if a.latest:
        print(max(rows, key=lambda r: r["saved_at"])["_abs"])
        return
    if a.list:
        print(f"{'saved':20s} {'ok':3s} {'secs':>6s} {'cover':>6s}  path")
        for r in sorted(rows, key=lambda r: r["saved_at"]):
            print(f"{r['saved_at']:20s} {'yes' if r['success'] == 'True' else 'no':3s} "
                  f"{float(r['duration_s'] or 0):6.1f} {r['final_coverage'] or '-':>6s}  {r['path']}")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["session"]), []).append(r)
    print(f"{'task':8s} {'session':22s} {'episodes':>8s} {'success':>8s} {'minutes':>8s} {'avg cover':>9s}")
    for (task, sess), rs in sorted(groups.items()):
        ok = sum(r["success"] == "True" for r in rs)
        mins = sum(float(r["duration_s"] or 0) for r in rs) / 60
        cov = [float(r["final_coverage"]) for r in rs if r["final_coverage"]]
        avg = f"{sum(cov) / len(cov):.2f}" if cov else "-"
        print(f"{task:8s} {sess:22s} {len(rs):8d} {ok:8d} {mins:8.1f} {avg:>9s}")
    print(f"\ntotal: {len(rows)} episodes, {sum(r['success'] == 'True' for r in rows)} successful")


if __name__ == "__main__":
    main()
