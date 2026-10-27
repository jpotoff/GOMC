#!/usr/bin/env python3
"""Sampling efficiency and averages for rigid 4-site water (OPC, TIP4P) runs.

Compares move sets (hybrid MC, single-molecule moves, multiparticle) on the
same system by how fast molecules decorrelate per unit of compute time.

Usage:
  water_sampling_report.py traj RUN_BOX_0.dcd SEC_PER_FRAME SKIP_FRAMES
                           [--box L] [--label NAME]
  water_sampling_report.py ener RUN.log SKIP_STEPS [--label NAME]

traj   centre-of-mass mean-squared displacement and the dipole-direction
       autocorrelation <u(0).u(t)> against compute time. SEC_PER_FRAME is
       the time between DCD frames:
           (time for the move, from the log's move-timing table)
           / RunSteps * DCDFreq
       Use wall-clock time to compare thread counts, CPU time to compare
       per-core efficiency. Compare methods by the time each needs to reach
       the same MSD or the same <u(0).u(t)>: ratios at a fixed time
       understate the difference, because both curves are sublinear.
ener   averages of the total energy (ENER_0) and pressure (STAT_0) after
       SKIP_STEPS, with block-average errors. Slowly relaxing systems (cold
       water) need long runs: the errors are only as good as the blocks are
       independent.

Assumes atoms are ordered O, H, H, M within each molecule and a cubic box
(default 24.64 A, test/input/Systems/CBMC/OPC_260K).
"""
import argparse
import struct

import numpy as np

MASS = np.array([15.999, 1.008, 1.008, 0.0])


def read_dcd(path):
    f = open(path, "rb")

    def record():
        (n,) = struct.unpack("i", f.read(4))
        data = f.read(n)
        f.read(4)
        return data

    hdr = record()
    has_cell = struct.unpack("i", hdr[44:48])[0] != 0
    record()  # title
    (natom,) = struct.unpack("i", record())
    frames = []
    while True:
        try:
            if has_cell:
                record()
            x = np.frombuffer(record(), dtype=np.float32)
            y = np.frombuffer(record(), dtype=np.float32)
            z = np.frombuffer(record(), dtype=np.float32)
        except (struct.error, ValueError):
            break
        if len(x) != natom:
            break
        frames.append(np.stack([x, y, z], axis=1).astype(np.float64))
    return np.array(frames)


def traj(path, spf, skip, box, label):
    def minimg(d):
        return d - box * np.round(d / box)

    xyz = read_dcd(path)[skip:]
    nf, natom, _ = xyz.shape
    mol = xyz.reshape(nf, natom // 4, 4, 3)
    # rebuild each molecule around its oxygen, then the mass-weighted centre
    rel = minimg(mol - mol[:, :, :1, :])
    com = mol[:, :, 0, :] + (rel * MASS[None, None, :, None]).sum(2) / MASS.sum()
    step = minimg(np.diff(com, axis=0))
    unwrapped = np.concatenate([com[:1], com[:1] + np.cumsum(step, axis=0)])
    u = 0.5 * (rel[:, :, 1, :] + rel[:, :, 2, :])  # O -> midpoint of the Hs
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    print(f"# {label}: {nf} frames after skipping {skip}, "
          f"{spf:.3f} s per frame")
    print(f"# {'lag':>5} {'time/s':>9} {'MSD/A^2':>10} {'<u(0).u(t)>':>12}")
    for lag in sorted({int(round(v)) for v in np.geomspace(1, nf // 2, 14)}):
        d = unwrapped[lag:] - unwrapped[:-lag]
        msd = (d ** 2).sum(2).mean()
        cu = (u[lag:] * u[:-lag]).sum(2).mean()
        print(f"  {lag:5d} {lag * spf:9.1f} {msd:10.4f} {cu:12.4f}")


def block(v, nb=10):
    v = np.asarray(v)
    bs = len(v) // nb
    m = v[: nb * bs].reshape(nb, bs).mean(1)
    return m.mean(), m.std(ddof=1) / np.sqrt(nb)


def ener(path, skip, label):
    e, p = [], []
    for line in open(path):
        f = line.split()
        if len(f) < 3:
            continue
        if f[0] == "ENER_0:" and int(f[1]) > skip:
            e.append(float(f[2]))
        if f[0] == "STAT_0:" and int(f[1]) > skip:
            p.append(float(f[2]))
    me, ee = block(e)
    print(f"{label}: {len(e)} samples  <E_total> = {me:.1f} +- {ee:.1f} K")
    if p:
        mp, ep = block(p)
        print(f"{label}: {len(p)} samples  <P> = {mp:.1f} +- {ep:.1f} bar")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("traj")
    t.add_argument("dcd")
    t.add_argument("sec_per_frame", type=float)
    t.add_argument("skip_frames", type=int)
    t.add_argument("--box", type=float, default=24.64)
    t.add_argument("--label", default=None)
    e = sub.add_parser("ener")
    e.add_argument("log")
    e.add_argument("skip_steps", type=int)
    e.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.mode == "traj":
        traj(a.dcd, a.sec_per_frame, a.skip_frames, a.box, a.label or a.dcd)
    else:
        ener(a.log, a.skip_steps, a.label or a.log)


if __name__ == "__main__":
    main()
