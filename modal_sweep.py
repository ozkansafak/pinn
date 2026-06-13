"""
modal_sweep.py — width sweep on Modal, laptop-disconnect-safe.

Usage:
    modal run modal_sweep.py            # launch sweep, then safe to close laptop
    modal run modal_sweep.py::download  # pull results to local disk when done
"""
import csv
import os
import sys
import modal

app   = modal.App("pinn-width-sweep")
volume = modal.Volume.from_name("pinn-sweep-results", create_if_missing=True)
VOL   = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_file("pinn.py",        remote_path="/root/pinn/pinn.py")
    .add_local_file("train_width.py", remote_path="/root/pinn/train_width.py")
    .add_local_file("optimizers.py",  remote_path="/root/pinn/optimizers.py")
)

N_F_BY_WIDTH = {
    4:   10_000,
    8:   10_000,
    16:  10_000,
    32:  10_000,
    64:  10_000,
    128: 40_000,
    256: 155_000,
    512: 600_000,
}


# ── Per-width training job (runs on a T4) ──────────────────────────────────────
@app.function(
    gpu="l4",
    image=image,
    volumes={VOL: volume},
    timeout=86_400,
)
def train_one(width: int, n_f: int, max_epochs: int = 60_000):
    import subprocess
    os.makedirs(VOL, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "/root/pinn/train_width.py",
            "--width",      str(width),
            "--n-f",        str(n_f),
            "--max-epochs", str(max_epochs),
            "--output",     f"{VOL}/width_{width}.csv",
            "--activation", "siren",
        ],
        capture_output=True, text=True, cwd="/root/pinn",
    )

    log = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
    with open(f"{VOL}/width_{width}.log", "w") as f:
        f.write(log)
    volume.commit()
    print(f"width={width} done.")


# ── Orchestrator (runs on Modal — laptop-safe) ─────────────────────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOL: volume},
    timeout=86_400,
)
def run_sweep():
    widths = sorted(N_F_BY_WIDTH)
    nf     = [N_F_BY_WIDTH[w] for w in widths]

    print(f"Launching {len(widths)} widths in parallel: {widths}")

    # Fan out all widths simultaneously; iterate to block until all finish
    for _ in train_one.map(widths, nf):
        pass

    # Merge per-width CSVs into one combined file
    all_rows = []
    for w in widths:
        path = f"{VOL}/width_{w}.csv"
        if os.path.exists(path):
            with open(path) as f:
                all_rows.extend(list(csv.DictReader(f)))

    if all_rows:
        out = f"{VOL}/width_sweep_v3.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda r: int(r["width"])))
        print(f"Saved merged CSV → {out}")

    volume.commit()
    print("Sweep complete!")


# ── Local trigger: run orchestrator detached (use: modal run --detach …) ───────
@app.local_entrypoint()
def main():
    print("Starting sweep orchestrator on Modal (detach-safe)...")
    run_sweep.remote()  # blocking — kept alive by --detach flag
    print("Sweep complete!")


# ── Targeted rerun for specific widths (use: modal run --detach …::rerun) ──────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOL: volume},
    timeout=86_400,
)
def run_rerun(widths: list):
    nf = [N_F_BY_WIDTH[w] for w in widths]
    print(f"Rerunning widths {widths} with N_f {nf}...")
    for _ in train_one.map(widths, nf):
        pass
    # Rebuild merged CSV
    all_rows = []
    for w in sorted(N_F_BY_WIDTH):
        path = f"{VOL}/width_{w}.csv"
        if os.path.exists(path):
            with open(path) as f:
                all_rows.extend(list(csv.DictReader(f)))
    if all_rows:
        out = f"{VOL}/width_sweep_v3.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda r: int(r["width"])))
        print(f"Saved merged CSV → {out}")
    volume.commit()
    print("Rerun complete!")


@app.local_entrypoint()
def rerun():
    # Spawn each width independently — no orchestrator in the middle to cancel
    for w in [256, 512]:
        train_one.spawn(w, N_F_BY_WIDTH[w])
        print(f"Spawned width={w} N_f={N_F_BY_WIDTH[w]}")


# ── Optimizer comparison: Adam vs ClampedCenteredAdam at w=64 ─────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOL: volume},
    timeout=86_400,
)
def run_opt_test():
    configs = [
        ("adam", 1.0),
        ("cca",  1.0),
        ("cca",  0.1),
        ("cca",  0.01),
    ]
    widths = [64] * len(configs)
    nf     = [10_000] * len(configs)
    opts   = [c[0] for c in configs]
    taus   = [c[1] for c in configs]

    print(f"Launching optimizer test: {configs}")
    for _ in train_one_opt.map(widths, nf, opts, taus):
        pass
    volume.commit()
    print("Optimizer test complete!")


@app.function(
    gpu="l4",
    image=image,
    volumes={VOL: volume},
    timeout=86_400,
)
def train_one_opt(width: int, n_f: int, opt_name: str, tau: float):
    import subprocess
    os.makedirs(VOL, exist_ok=True)
    tag = f"{opt_name}_tau{tau}" if opt_name == "cca" else opt_name
    result = subprocess.run(
        [
            sys.executable, "/root/pinn/train_width.py",
            "--width",      str(width),
            "--n-f",        str(n_f),
            "--max-epochs", "60000",
            "--output",     f"{VOL}/opt_test_{tag}.csv",
            "--activation", "siren",
            "--optimizer",  opt_name,
            "--tau",        str(tau),
        ],
        capture_output=True, text=True, cwd="/root/pinn",
    )
    log = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
    with open(f"{VOL}/opt_test_{tag}.log", "w") as f:
        f.write(log)
    volume.commit()
    print(f"opt={opt_name} tau={tau} done.")


@app.local_entrypoint()
def run_opt(opt: str = "adam", tau: float = 1.0):
    """Launch a single optimizer run. Use --detach to keep it alive after disconnect.
    Examples:
        modal run --detach modal_sweep.py::run_opt --opt adam
        modal run --detach modal_sweep.py::run_opt --opt cca --tau 0.1
    """
    train_one_opt.remote(64, 10_000, opt, tau)  # last triggered — survives --detach


# ── Download results from Volume to local disk ─────────────────────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOL: volume},
)
def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


@app.local_entrypoint()
def download():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # Combined CSV
    csv_content = _read_file.remote(f"{VOL}/width_sweep_v3.csv")
    with open("results/width_sweep_v3.csv", "w") as f:
        f.write(csv_content)
    print("Downloaded → results/width_sweep_v3.csv")

    # Per-width logs
    for w in sorted(N_F_BY_WIDTH):
        log = _read_file.remote(f"{VOL}/width_{w}.log")
        path = f"results/logs/width_{w}_siren_modal.log"
        with open(path, "w") as f:
            f.write(log)
        print(f"Downloaded → {path}")
