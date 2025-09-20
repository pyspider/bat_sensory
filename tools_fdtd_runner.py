#!/usr/bin/env python3
"""
tools/fdtd_runner.py

Lightweight Python interface to prepare a case for the WE-FDTD executable,
run it (optionally via mpirun), and parse output waveform CSV(s).

Environment:
- WE_FDTD_BIN: path to compiled WE-FDTD binary (default "./WE-FDTD")
- MPIRUN: path to mpirun (default "mpirun")

This module is dependency-free (stdlib only).
"""
from __future__ import annotations
import os
import shutil
import subprocess
import csv
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

WE_FDTD_BIN = os.environ.get("WE_FDTD_BIN", "./WE-FDTD")
MPIRUN = os.environ.get("MPIRUN", "mpirun")


class FDTDError(RuntimeError):
    pass


def write_input_dat(target_dir: str, params: Dict) -> str:
    """
    Write a minimal input.dat into target_dir based on params.
    The template is aligned with ReadModel() expectations from WE-FDTD.h.
    """
    p = Path(target_dir)
    p.mkdir(parents=True, exist_ok=True)
    fn = p / "input.dat"
    lines = []
    lines.append("1\n")  # dummy header (original ReadModel reads this)
    lines.append(f"{params.get('iCell',0)} {params.get('Scheme',1)} {params.get('Boundary',0)}\n")
    lines.append(f"{params.get('Nx',200)} {params.get('Ny',200)} {params.get('CellName','cell.dat')}\n")
    lines.append(f"{params.get('Ref0',0.0)} {params.get('Ref1',0.0)} {params.get('Ref2',0.0)} {params.get('Ref3',0.0)} {params.get('aref',0.0)}\n")
    lines.append(f"{params.get('cfl',0.70710678)} {params.get('dl',0.001)} {params.get('c0',343.0)}\n")
    lines.append(f"{params.get('Nt',2000)}\n")
    lines.append(f"{params.get('iSource',0)} {params.get('SPosName','none')}\n")
    s = params.get('Src', (50,50,1.0,0,0,0))
    lines.append(f"{int(s[0])} {int(s[1])} {s[2]} {s[3]} {s[4]} {s[5]}\n")
    lines.append(f"{params.get('freq',0)} {params.get('Nd',0)} {params.get('SrcName','src.wav')}\n")
    lines.append(f"{params.get('iReceiver',0)} {params.get('RPosName','none')}\n")
    r = params.get('Rcv', (50,80,1.0,0,0,0))
    lines.append(f"{int(r[0])} {int(r[1])} {r[2]} {r[3]} {r[4]} {r[5]}\n")
    lines.append(f"{params.get('iplane',0)} {params.get('iwave',1)}\n")
    lines.append(f"{params.get('iptime',0)} {params.get('istx',1)} {params.get('isty',1)} {params.get('ipts',0)} {params.get('ipte',0)}\n")
    lines.append(f"{params.get('Ngpu',1)} {params.get('GpuId',0)}\n")

    with open(fn, "w") as fh:
        fh.writelines(lines)
    return str(fn)


def run_fdtd_case(case_dir: str, mpirun_np: int = 0, timeout: Optional[int] = None) -> Tuple[int, str]:
    """
    Run WE-FDTD in case_dir. If mpirun_np > 1 will call mpirun -np mpirun_np WE_FDTD_BIN.
    Returns (returncode, log_path). Raises FDTDError on basic failure to spawn.
    """
    case_dir = os.path.abspath(case_dir)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(case_dir)

    if mpirun_np and mpirun_np > 1:
        cmd = [MPIRUN, "-np", str(mpirun_np), WE_FDTD_BIN]
    else:
        cmd = [WE_FDTD_BIN]

    logpath = os.path.join(case_dir, "we_fdtd_run.log")
    with open(logpath, "w") as logf:
        try:
            proc = subprocess.run(cmd, cwd=case_dir, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return proc.returncode, logpath
        except FileNotFoundError as e:
            raise FDTDError(f"Failed to start WE-FDTD executable ({WE_FDTD_BIN}): {e}")
        except subprocess.TimeoutExpired:
            logf.write("\n*** TIMEOUT ***\n")
            raise FDTDError("WE-FDTD run timed out (see log: %s)" % logpath)


def read_wave_csv(case_dir: str) -> List[Tuple[float, float]]:
    """
    Parse wave.csv or recv*.csv in case_dir. Return list of (timestep_index, amplitude).
    The WE-FDTD output format can vary; this attempts common cases.
    """
    candidates = [
        os.path.join(case_dir, "wave.csv"),
    ]
    # also add any file matching recv*.csv
    for name in os.listdir(case_dir):
        if name.startswith("recv") and name.endswith(".csv"):
            candidates.append(os.path.join(case_dir, name))

    for p in candidates:
        if os.path.exists(p):
            data = []
            with open(p, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    # try numeric parsing
                    try:
                        if len(row) >= 2:
                            t = float(row[0])
                            v = float(row[1])
                        else:
                            # single column -> treat as amplitude, index as time
                            t = float(i)
                            v = float(row[0])
                        data.append((t, v))
                    except Exception:
                        # skip malformed lines
                        continue
            if data:
                return data
    raise FileNotFoundError("No wave CSV found in %s (looked at: %s)" % (case_dir, ", ".join(candidates)))


def run_and_get_wave(params: Dict, scratch_base: Optional[str] = None, mpirun_np: int = 0, timeout: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    High-level convenience: create temp case dir (or under scratch_base), copy support files if given,
    write input.dat, run WE-FDTD, and return waveform.
    params may contain:
      - support_dir: path with support files to copy (cell.dat, SourceWave.csv...)
      - other fields used by write_input_dat (Nx, Ny, Nt, Src, Rcv, etc.)
    """
    base = scratch_base or tempfile.mkdtemp(prefix="fdtd_case_")
    case_name = params.get("case_name", "case")
    case_dir = os.path.join(base, case_name)
    os.makedirs(case_dir, exist_ok=True)

    # copy support files
    sup = params.get("support_dir")
    if sup:
        sup = os.path.abspath(sup)
        if not os.path.isdir(sup):
            raise FileNotFoundError(f"support_dir not found: {sup}")
        for f in os.listdir(sup):
            src = os.path.join(sup, f)
            dst = os.path.join(case_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    write_input_dat(case_dir, params)

    rc, log = run_fdtd_case(case_dir, mpirun_np=mpirun_np, timeout=timeout)
    if rc != 0:
        raise FDTDError(f"WE-FDTD failed (rc={rc}); see log: {log}")

    wave = read_wave_csv(case_dir)
    return wave


def _demo():
    """Self-test demo that prepares a tiny case and prints waveform length."""
    params = {
        "case_name": "demo1",
        "Nx": 120, "Ny": 120,
        "dl": 0.005,
        "Nt": 200,
        "iSource": 0,
        "Src": (60, 30, 1.0, 0, 0, 0),
        "iReceiver": 0,
        "Rcv": (60, 90, 1.0, 0, 0, 0),
        "Ngpu": 1, "GpuId": 0,
        "freq": 0,
    }
    base = "./fdtd_cases_demo"
    os.makedirs(base, exist_ok=True)
    try:
        w = run_and_get_wave(params, scratch_base=base, mpirun_np=1, timeout=60)
        print("Waveform samples:", len(w))
    except Exception as e:
        print("Demo failed:", e)
        print("Make sure WE-FDTD binary exists and environment variables are set:")
        print("  WE_FDTD_BIN=", WE_FDTD_BIN)


if __name__ == "__main__":
    _demo()