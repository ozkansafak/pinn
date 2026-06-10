"""Parse existing width_*.log files into per-width loss curve CSVs."""
import re
import csv
import os
import glob

LOG_DIR   = "results/logs"
OUT_DIR   = "results/loss_curves"
os.makedirs(OUT_DIR, exist_ok=True)

pattern = re.compile(
    r"epoch\s+(\d+)\s+\|\s+train_pde\s+([\d.e+\-]+)\s+\|\s+train_bc\s+([\d.e+\-]+)"
    r"\s+\|\s+eval_pde\s+([\d.e+\-]+)\s+\|\s+lr\s+([\d.e+\-]+)"
)

for log_path in sorted(glob.glob(f"{LOG_DIR}/width_*.log")):
    width = os.path.basename(log_path).replace("width_", "").replace(".log", "")
    out_path = f"{OUT_DIR}/width_{width}.csv"

    rows = []
    with open(log_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                rows.append({
                    "epoch":      int(m.group(1)),
                    "train_pde":  m.group(2),
                    "train_bc":   m.group(3),
                    "eval_pde":   m.group(4),
                    "lr":         m.group(5),
                })

    if not rows:
        print(f"  width={width}: no epoch lines found, skipping")
        continue

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_pde", "train_bc", "eval_pde", "lr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  width={width:4s}: {len(rows)} rows → {out_path}")
