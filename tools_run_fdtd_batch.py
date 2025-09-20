#!/usr/bin/env python3
"""
tools/run_fdtd_batch.py

Simple batch launcher to run multiple WE-FDTD cases in parallel.

cases file format: each line:
  <case_name> <case_dir>
If only one token on a line, case_dir defaults to current directory.
"""
from __future__ import annotations
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def run_case(exec_path, case_name, case_dir, mpirun_np, env=None):
    outdir = Path("outputs") / case_name
    outdir.mkdir(parents=True, exist_ok=True)
    # copy input files from case_dir into outdir (if case_dir exists)
    if case_dir and os.path.isdir(case_dir):
        for f in os.listdir(case_dir):
            src = os.path.join(case_dir, f)
            dst = os.path.join(outdir, f)
            if os.path.isfile(src):
                try:
                    subprocess.run(["cp", src, dst], check=True)
                except Exception:
                    pass
    cmd = []
    if mpirun_np and mpirun_np > 1:
        cmd = ["mpirun", "-np", str(mpirun_np), exec_path]
    else:
        cmd = [exec_path]

    print("Running case", case_name, "cmd:", " ".join(cmd))
    log = outdir / "run.log"
    with open(log, "w") as logf:
        proc = subprocess.run(cmd, cwd=outdir, stdout=logf, stderr=subprocess.STDOUT)
    return case_name, proc.returncode, str(log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="cases list file")
    parser.add_argument("--exec", default="./WE-FDTD", help="path to binary")
    parser.add_argument("--np", type=int, default=1, help="mpirun -np (use 0 or 1 to run directly)")
    parser.add_argument("--parallel", type=int, default=1, help="number of parallel case processes")
    args = parser.parse_args()

    exec_path = args.exec
    mpirun_np = args.np if args.np > 1 else 0

    cases = []
    with open(args.cases, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) == 1:
                cases.append((parts[0], "."))
            else:
                cases.append((parts[0], parts[1]))

    os.makedirs("outputs", exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_case, exec_path, name, dirn, mpirun_np) for name, dirn in cases]
        for f in as_completed(futures):
            name, rc, log = f.result()
            results.append((name, rc, log))
            print("Case:", name, "rc=", rc, "log=", log)

    print("Summary:")
    for name, rc, log in results:
        print(name, rc, log)

if __name__ == "__main__":
    main()