#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ftplib
import logging
import os
import socket
import time
from pathlib import Path, PurePosixPath

FTP_HOST = "ftp.awi.de"
REMOTE_BASE = "/sea_ice/projects/cci/crdp/v4p0"
DEFAULT_DEST = Path("/g/data/gv90/da1339/SeaIce/AWI")
LOGGER = logging.getLogger("download_AWI_SIT")


def _list(ftp: ftplib.FTP, path: str):
    try:
        return [(n, f.get("type"), int(f["size"]) if f.get("size","").isdigit() else None)
                for n, f in ftp.mlsd(path, facts=["type","size"])
                if n not in {".",".."}]
    except Exception:
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        out = []
        for line in lines:
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                continue
            kind = "dir" if parts[0].startswith("d") else "file"
            size = int(parts[4]) if kind == "file" and parts[4].isdigit() else None
            out.append((parts[-1], kind, size))
        return out


def _download(ftp, remote, local, expected):
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and (expected is None or local.stat().st_size == expected):
        return "skip"
    part = local.with_suffix(local.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    for attempt in range(1, 6):
        try:
            with open(part, "ab" if offset else "wb") as fh:
                ftp.retrbinary(
                    f"RETR {remote}", fh.write, blocksize=1024*256,
                    rest=offset if offset else None
                )
            if expected is not None and part.stat().st_size != expected:
                raise IOError("size mismatch")
            part.replace(local)
            return "download"
        except (OSError, socket.timeout, ftplib.Error) as exc:
            LOGGER.warning("Attempt %d failed for %s: %s", attempt, remote, exc)
            time.sleep(2 * attempt)
            offset = part.stat().st_size if part.exists() else 0
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--hemispheres", default="nh,sh")
    ap.add_argument("--levels", default="l2p_release,l3cp_release")
    ap.add_argument("--sensors", default="envisat,cryosat2,sentinel3a,sentinel3b")
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hems = {x.strip().lower() for x in args.hemispheres.split(",")}
    levels = {x.strip().lower() for x in args.levels.split(",")}
    sensors = {x.strip().lower() for x in args.sensors.split(",")}
    ftp = ftplib.FTP(FTP_HOST, timeout=180)
    ftp.login("anonymous", os.environ.get("FTP_PASS", "anonymous@"))
    ftp.set_pasv(True)
    stack = [REMOTE_BASE]
    n_down = n_skip = n_fail = 0

    def allowed(path):
        parts = [p.lower() for p in PurePosixPath(path).parts]
        if {"l2p_release","l3cp_release"}.intersection(parts) and not set(parts).intersection(levels):
            return False
        if {"nh","sh"}.intersection(parts) and not set(parts).intersection(hems):
            return False
        if {"envisat","cryosat2","sentinel3a","sentinel3b"}.intersection(parts) and not set(parts).intersection(sensors):
            return False
        yrs = [int(p) for p in parts if p.isdigit() and len(p) == 4]
        if yrs:
            y = yrs[-1]
            if args.year_min is not None and y < args.year_min:
                return False
            if args.year_max is not None and y > args.year_max:
                return False
        return True

    while stack:
        rdir = stack.pop()
        try:
            entries = _list(ftp, rdir)
        except Exception as exc:
            LOGGER.warning("Cannot list %s: %s", rdir, exc)
            continue

        for name, kind, size in entries:
            rp = f"{rdir}/{name}"
            if kind == "dir":
                if allowed(rp):
                    stack.append(rp)
                continue
            if not name.endswith(".nc") or not allowed(rp):
                continue
            rel = PurePosixPath(rdir).relative_to(PurePosixPath(REMOTE_BASE))
            lp = args.dest / Path(str(rel)) / name
            result = _download(ftp, rp, lp, size)
            n_skip += result == "skip"
            n_down += result == "download"
            n_fail += result == "fail"
            if result == "download":
                LOGGER.info("Downloaded %s", lp)

    LOGGER.info("Summary downloaded=%d skipped=%d failed=%d", n_down, n_skip, n_fail)
    if n_fail:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
