#!/usr/bin/env python3
"""Sampling efficiency and averages for GOMC runs, any molecule types.

Compares move sets (hybrid MC, single-molecule moves, multiparticle, ...) on
the same system by how fast molecules decorrelate per unit of compute time.

Usage:
  sampling_report.py traj RUN_BOX_0.dcd MERGED.psf SEC_PER_FRAME SKIP_FRAMES
                     [--kind RES ...] [--vector RES:A:B[,C...] ...]
                     [--label NAME]
  sampling_report.py ener RUN.log SKIP_STEPS [--box B] [--label NAME]

traj   for each molecule kind (PSF residue name), against compute time:
         MSD     centre-of-mass mean-squared displacement, A^2
         <u.u>   orientation autocorrelation <u(0).u(t)> of a unit vector
                 fixed in the molecule (see --vector)
         C_dih   dihedral autocorrelation <dcos(0) dcos(t)> / <dcos^2> of
                 the heavy-atom dihedrals (dcos = cos(phi) - <cos(phi)>,
                 the mean taken over all of the kind's heavy-atom dihedrals
                 and frames): how fast conformations (trans/gauche)
                 decorrelate. Shown for kinds with heavy-atom dihedrals in
                 the PSF.
       SEC_PER_FRAME is the time between DCD frames:
           (time for the moves, from the log's move-timing table)
           / RunSteps * DCDFreq
       Use wall-clock time to compare thread counts, CPU time to compare
       per-core efficiency. Compare methods by the time each needs to reach
       the same MSD, <u.u> or C_dih: ratios at a fixed time understate the
       difference, because the curves are sublinear.
ener   averages of the total energy (ENER_B) and pressure (STAT_B) after
       SKIP_STEPS, with block-average errors. Slowly relaxing systems need
       long runs: the errors are only as good as the blocks are independent.
       GOMC updates the pressure every PressureCalc steps, so set that no
       larger than ConsoleFreq.

Molecules are consecutive atoms with the same segment and residue ID in the
PSF, as GOMC defines them; use the merged PSF GOMC writes, which matches the
DCD atom for atom. Masses come from the PSF (massless sites such as the OPC
M site carry no weight in the centre of mass).

--vector RES:A:B[,C...]  orientation vector of kind RES from atom A to the
       centroid of atoms B, C, ... (PSF atom names). For water,
       OPC:OW:HW1,HW2 is the dipole direction. Without it, a kind uses the
       vector between its two atoms farthest apart in the first frame
       (end to end for a chain). Single-atom kinds have no orientation.
--kind RES   analyse only these kinds (default: all).

Boxes: the box edge is read from each frame's unit cell, so NPT works.
Boxes must be orthogonal. In GEMC and GCMC each box's DCD holds every atom,
with molecules outside the box at the origin; a molecule counts for a lag
only if it is in the box in every frame of that window. A molecule that
leaves and returns between two consecutive frames cannot be detected, so
write frames often compared with the swap rate.
"""
import argparse
import struct
import sys
from collections import OrderedDict

import numpy as np


# ------------------------------------------------------------------ input --

def read_dcd(path):
    """Coordinates (frames, atoms, 3) and box edges (frames, 3)."""
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
    frames, boxes = [], []
    while True:
        try:
            cell = np.frombuffer(record(), dtype=np.float64) if has_cell else None
            x = np.frombuffer(record(), dtype=np.float32)
            y = np.frombuffer(record(), dtype=np.float32)
            z = np.frombuffer(record(), dtype=np.float32)
        except (struct.error, ValueError):
            break
        if len(x) != natom:
            break
        frames.append(np.stack([x, y, z], axis=1).astype(np.float64))
        if cell is not None:
            # NAMD layout: a, cos(gamma), b, cos(beta), cos(alpha), c
            if np.any(np.abs(cell[[1, 3, 4]]) > 1e-6):
                sys.exit("non-orthogonal box: not supported")
            boxes.append([cell[0], cell[2], cell[5]])
    if not has_cell:
        sys.exit(f"{path} has no unit cell; GOMC writes one per frame")
    return np.array(frames), np.array(boxes)


def read_psf(path):
    """Atom names, residue names, masses, molecule index per atom, and
    dihedrals (4-tuples of atom indices)."""
    lines = open(path).read().splitlines()
    names, resn, mass, molkey = [], [], [], []
    dih = []
    i = 0
    while i < len(lines):
        f = lines[i].split()
        if len(f) >= 2 and f[1].startswith("!NATOM"):
            n = int(f[0])
            for line in lines[i + 1:i + 1 + n]:
                a = line.split()
                # id segid resid resname name type charge mass
                molkey.append((a[1], a[2]))
                resn.append(a[3])
                names.append(a[4])
                mass.append(float(a[7]))
            i += n
        elif len(f) >= 2 and f[1].startswith("!NPHI"):
            n = int(f[0])
            vals = []
            j = i + 1
            while len(vals) < 4 * n:
                vals += [int(v) - 1 for v in lines[j].split()]
                j += 1
            dih = [tuple(vals[k:k + 4]) for k in range(0, 4 * n, 4)]
            i = j - 1
        i += 1
    mol = np.zeros(len(names), dtype=int)
    m = 0
    for k in range(1, len(names)):
        if molkey[k] != molkey[k - 1]:
            m += 1
        mol[k] = m
    return (np.array(names), np.array(resn), np.array(mass), mol,
            np.array(dih, dtype=int).reshape(-1, 4))


# --------------------------------------------------------------- analysis --

def minimg(d, box):
    return d - box * np.round(d / box)


def lags(nf):
    return sorted({int(round(v)) for v in np.geomspace(1, max(1, nf // 2), 14)})


def window_ok(present, lag):
    """ok[t, m]: molecule m in the box in every frame t .. t + lag."""
    absent = np.concatenate([np.zeros((1, present.shape[1]), dtype=int),
                             np.cumsum(~present, axis=0)])
    return (absent[lag + 1:] - absent[:-lag - 1]) == 0


def dihedral_cos(xyz, d, box):
    """cos(phi) for dihedrals d (k, 4), frames x k."""
    p = [xyz[:, d[:, n], :] for n in range(4)]
    b1 = minimg(p[1] - p[0], box[:, None, :])
    b2 = minimg(p[2] - p[1], box[:, None, :])
    b3 = minimg(p[3] - p[2], box[:, None, :])
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    c = (n1 * n2).sum(2) / (np.linalg.norm(n1, axis=2) *
                            np.linalg.norm(n2, axis=2))
    return np.clip(c, -1.0, 1.0)


def analyse_kind(res, idx, xyz, box, names, mass, dih, vec_spec, spf, label):
    nf = xyz.shape[0]
    nper = len(idx[0])
    nmol = len(idx)
    first = np.array([ix[0] for ix in idx])
    atoms = np.array(idx)  # nmol x nper
    mol = xyz[:, atoms, :]  # frames x nmol x nper x 3
    # Molecules outside this box are written with every atom at the origin.
    present = np.any(mol != 0.0, axis=(2, 3))
    bx = box[:, None, None, :]
    rel = minimg(mol - mol[:, :, :1, :], bx)
    w = mass[atoms[0]]
    if w.sum() > 0:
        com = mol[:, :, 0, :] + (rel * w[None, None, :, None]).sum(2) / w.sum()
    else:
        com = mol[:, :, 0, :]
    step = minimg(np.diff(com, axis=0), box[1:, None, :])
    step[~(present[1:] & present[:-1])] = 0.0
    # Unwrapping assumes a molecule moves less than half a box per frame.
    far = (np.abs(step) > 0.35 * box[1:, None, :]).any(2).mean()
    unwrapped = np.concatenate([com[:1], com[:1] + np.cumsum(step, axis=0)])

    # Orientation vector, in the molecule's own atom numbering.
    u = None
    vec_desc = None
    if nper > 1:
        nm = list(names[atoms[0]])
        if vec_spec is not None:
            a, bs = vec_spec
            try:
                ia, ib = nm.index(a), [nm.index(b) for b in bs]
            except ValueError:
                sys.exit(f"--vector {res}:{a}:{','.join(bs)}: atom names "
                         f"are {' '.join(nm)}")
            vec_desc = f"{a} -> {'+'.join(bs)}"
        else:
            m0 = next((m for m in range(nmol) if present[0, m]), 0)
            r = rel[0, m0]
            d2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(2)
            ia, ibb = np.unravel_index(np.argmax(d2), d2.shape)
            ib = [ibb]
            vec_desc = f"{nm[ia]} -> {nm[ibb]} (farthest pair)"
        u = rel[:, :, ib, :].mean(2) - rel[:, :, ia, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            u /= np.linalg.norm(u, axis=2, keepdims=True)
        u = np.where(present[:, :, None], u, 0.0)

    # Heavy-atom dihedrals of this kind, as (molecule, dihedral) pairs.
    cosd = None
    ndih = 0
    if len(dih):
        member = np.full(len(names), -1)
        member[atoms.ravel()] = np.repeat(np.arange(nmol), nper)
        inkind = (member[dih] >= 0).all(1) & (member[dih[:, 0]][:, None] ==
                                              member[dih]).all(1)
        heavy = (mass[dih] > 4.0).all(1)
        dk = dih[inkind & heavy]
        if len(dk):
            ndih = len(dk) // nmol
            with np.errstate(invalid="ignore", divide="ignore"):
                cosd = dihedral_cos(xyz, dk, box)  # frames x (nmol * ndih)
            dmol = member[dk[:, 0]]
            cosd = np.where(present[:, dmol], cosd, 0.0)

    print(f"# {label}  kind {res}: {nmol} molecules x {nper} atoms, "
          f"{nf} frames, {spf:.4g} s per frame")
    if vec_desc:
        print(f"#   orientation vector {vec_desc}")
    if cosd is not None:
        print(f"#   {ndih} heavy-atom dihedrals per molecule")
    if far > 0.001:
        print(f"#   WARNING: {100 * far:.1f}% of molecule steps between frames "
              f"exceed 0.35 box lengths; MSD is unreliable (write frames "
              f"more often, or ignore MSD for this box)")
    if not present.all():
        print(f"#   in this box: {present.sum(1).min()}-{present.sum(1).max()}"
              f" molecules per frame")
    hdr = f"# {'lag':>5} {'time/s':>10} {'N':>6} {'MSD/A^2':>10}"
    if u is not None:
        hdr += f" {'<u.u>':>8}"
    if cosd is not None:
        hdr += f" {'C_dih':>8}"
    print(hdr)

    if cosd is not None:
        # Fluctuations about the kind's mean: a per-dihedral time average
        # over a short run would bias C_dih negative at long lags.
        pd = present[:, dmol]
        mean = cosd[pd].mean() if pd.any() else 0.0
        dc = np.where(pd, cosd - mean, 0.0)
        var = (dc ** 2).sum() / max(pd.sum(), 1)

    for lag in lags(nf):
        if lag >= nf:
            break
        ok = window_ok(present, lag)  # (nf - lag) x nmol
        n = ok.sum()
        if n == 0:
            continue
        d = unwrapped[lag:] - unwrapped[:-lag]
        msd = ((d ** 2).sum(2) * ok).sum() / n
        line = f"  {lag:5d} {lag * spf:10.4g} {n / (nf - lag):6.0f} {msd:10.4f}"
        if u is not None:
            cu = ((u[lag:] * u[:-lag]).sum(2) * ok).sum() / n
            line += f" {cu:8.4f}"
        if cosd is not None:
            okd = ok[:, dmol]
            cd = ((dc[lag:] * dc[:-lag]) * okd).sum() / max(okd.sum(), 1)
            line += f" {cd / var if var > 0 else 0.0:8.4f}"
        print(line)
    print()


def traj(a):
    xyz, box = read_dcd(a.dcd)
    names, resn, mass, mol, dih = read_psf(a.psf)
    if xyz.shape[1] != len(names):
        sys.exit(f"DCD has {xyz.shape[1]} atoms, PSF {len(names)}: use the "
                 "merged PSF from the same run")
    xyz, box = xyz[a.skip_frames:], box[a.skip_frames:]
    if len(xyz) < 2:
        sys.exit("fewer than 2 frames after skipping")

    vecs = {}
    for v in a.vector or []:
        try:
            res, at, bs = v.split(":")
        except ValueError:
            sys.exit(f"--vector {v}: expected RES:A:B[,C...]")
        vecs[res] = (at, bs.split(","))

    kinds = OrderedDict()
    for m in range(mol.max() + 1):
        ix = np.nonzero(mol == m)[0]
        kinds.setdefault(resn[ix[0]], []).append(ix)
    label = a.label or a.dcd
    for res, idx in kinds.items():
        if a.kind and res not in a.kind:
            continue
        if len({len(ix) for ix in idx}) != 1:
            print(f"# kind {res}: molecules differ in atom count, skipped")
            continue
        analyse_kind(res, idx, xyz, box, names, mass, dih, vecs.get(res),
                     a.sec_per_frame, label)


def block(v, nb=10):
    v = np.asarray(v)
    bs = len(v) // nb
    m = v[: nb * bs].reshape(nb, bs).mean(1)
    return m.mean(), m.std(ddof=1) / np.sqrt(nb)


def ener(a):
    e, p = [], []
    etag, ptag = f"ENER_{a.box}:", f"STAT_{a.box}:"
    for line in open(a.log):
        f = line.split()
        if len(f) < 3:
            continue
        if f[0] == etag and int(f[1]) > a.skip_steps:
            e.append(float(f[2]))
        if f[0] == ptag and int(f[1]) > a.skip_steps:
            p.append(float(f[2]))
    label = a.label or a.log
    if not e:
        sys.exit(f"no {etag} lines after step {a.skip_steps}")
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
    t.add_argument("psf")
    t.add_argument("sec_per_frame", type=float)
    t.add_argument("skip_frames", type=int)
    t.add_argument("--kind", nargs="+")
    t.add_argument("--vector", action="append")
    t.add_argument("--label")
    e = sub.add_parser("ener")
    e.add_argument("log")
    e.add_argument("skip_steps", type=int)
    e.add_argument("--box", type=int, default=0)
    e.add_argument("--label")
    a = ap.parse_args()
    traj(a) if a.mode == "traj" else ener(a)


if __name__ == "__main__":
    main()
