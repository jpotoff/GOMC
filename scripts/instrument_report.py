#!/usr/bin/env python3
"""Summarize and interpret a GOMC <OutputName>_instrument.dat file.

Usage: instrument_report.py run_instrument.dat [--all] [--schedule N]

For every move, box, and molecule kind it reports the acceptance, why
attempts were rejected, how far rejected attempts were from threshold, and
whether the CBMC trial counts are adequate, followed by a diagnosis. With
NeMTMC data from a GCMC run it also prints the dissipated work per lambda
window and, with --schedule N, an N-window schedule that spaces the windows
at equal thermodynamic length.

Pure Python (no numpy), so it runs wherever GOMC runs.
"""
import argparse
import math
import sys

REASONS = ['accepted', 'metropolis', 'overlap_first', 'overlap_later',
           'zerow_first', 'zerow_later', 'overlap', 'lowweight_first',
           'lowweight_later', 'early']


def read_sections(path):
    sec, cur = {}, None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            cur = line[1:-1]
            sec[cur] = []
        elif cur is not None:
            sec[cur].append(line.split())
    return sec


def hist_quantile(counts, lo, width, q):
    tot = sum(counts)
    if tot == 0:
        return float('nan')
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if acc >= q * tot:
            return lo + (i + 0.5) * width
    return lo + (len(counts) - 0.5) * width


def frac_ge(counts, lo, width, x):
    tot = sum(counts)
    if tot == 0:
        return float('nan')
    return sum(c for i, c in enumerate(counts) if lo + i * width >= x) / tot


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report_moves(sec, show_all):
    rej = {tuple(r[:4]): r for r in sec.get('rejection', [])}
    lnacc = {tuple(r[:4]): r for r in sec.get('lnacc', [])}
    stages = {}
    for r in sec.get('stages', []):
        stages.setdefault(tuple(r[:4]), {})[r[4]] = r
    widom = {tuple(r[:4]): r for r in sec.get('widom', [])}
    positions = {}
    for r in sec.get('stage_position', []):
        positions.setdefault(tuple(r[:4]), []).append(r)

    for key in sorted(rej):
        r = rej[key]
        move, box, kind, nested = key
        n = int(r[4])
        cnt = dict(zip(REASONS, map(int, r[5:5 + len(REASONS)])))
        is_cbmc = key in stages
        if n == 0 or (not show_all and not is_cbmc and move not in
                      ('Nonequilibrium_Molecule_Transfer',)):
            continue
        where = f'{move} box {box} {kind}' + (' (inside NeMTMC)' if nested == '1'
                                               else '')
        print(f'\n=== {where}: {n} attempts, {pct(cnt["accepted"], n):.2f}% '
              f'accepted')
        tried = n - cnt['early']
        if cnt['early']:
            print(f'  rejected before trial generation: '
                  f'{pct(cnt["early"], n):.1f}%  (see [early])')
        first = (cnt['overlap_first'] + cnt['zerow_first'] +
                 cnt['lowweight_first'])
        later = (cnt['overlap_later'] + cnt['zerow_later'] +
                 cnt['lowweight_later'])
        struct = first + later + cnt['overlap']
        print(f'  rejections, % of attempts reaching trial generation:')
        for name in REASONS[1:-1]:
            if cnt[name]:
                print(f'    {name:14s} {pct(cnt[name], tried):6.2f}%')

        # where the acceptance ratio sits for attempts that had a finite one
        diag = []
        if key in lnacc:
            lr = lnacc[key]
            counts = list(map(int, lr[5:]))
            lo = -50.0
            if sum(counts):
                med = hist_quantile(counts, lo, 1.0, 0.5)
                near = frac_ge(counts, lo, 1.0, -3.0)
                far = 1.0 - frac_ge(counts, lo, 1.0, -20.0)
                print(f'  ln(acceptance ratio), finite cases: median {med:.1f},'
                      f' {100 * near:.0f}% above -3, {100 * far:.0f}% below -20'
                      f' (bin <= -50 holds {counts[0]})')
                if med > -5:
                    diag.append('finite-ratio attempts are near threshold: '
                                'tuning (more trials, better schedule) pays off')
                elif med < -20:
                    diag.append('finite-ratio attempts are orders of magnitude '
                                'below threshold: tuning will not rescue this '
                                'move; a different method is needed')

        if tried and struct > 0.5 * tried:
            if first > 2 * later:
                diag.append('most attempts die at first-bead placement: a '
                            'cavity problem. More trials help little; consider '
                            'CFCMC, MEMC, NeMTMC, or cavity-biased insertion')
            elif later > 2 * first:
                diag.append('most attempts die after placement, during growth: '
                            'a conformational problem. More or better CBMC '
                            'trials (CBMC_Nth, CBMC_Ang, CBMC_Dih) or recoil '
                            'growth should help')
            else:
                diag.append('structural failures split between placement and '
                            'growth')

        for st in ('first', 'later'):
            s = stages.get(key, {}).get(st)
            if not s:
                continue
            ns, nz, nover = int(s[5]), int(s[6]), int(s[7])
            mover, mbest, f90 = float(s[8]), float(s[9]), float(s[10])
            mlnw = float(s[11])
            print(f'  {st:5s} stage: {ns} trial sets, {pct(nz, ns):.1f}% all '
                  f'zero ({pct(nover, ns):.1f}% all overlap), mean overlap '
                  f'fraction {mover:.2f}, mean best-trial share {mbest:.2f}, '
                  f'{100 * f90:.0f}% of sets with best share > 0.9, mean '
                  f'ln(mean trial weight) {mlnw:.2f}')
            if f90 > 0.5:
                trials = 'CBMC_First' if st == 'first' else 'CBMC_Nth'
                diag.append(f'{st}-stage trial sets are dominated by one '
                            f'trial: {trials} is too small')
        pos = positions.get(key)
        if pos and any(int(r[12]) for r in pos):
            print('  by growth position: ln(mean trial weight) new, old, '
                  'paired new-old, sets with best share > 0.9')
            diffs = []
            for r in pos:
                d = float(r[13]) if int(r[12]) else float('nan')
                diffs.append(d)
                print(f'    pos {int(r[4]):2d} {r[5]:5s} new {float(r[9]):7.2f}'
                      f'  old {float(r[11]):7.2f}  diff {d:7.2f}'
                      f'  best>0.9 {100 * float(r[8]):4.1f}%')
            lost = [x for x in diffs if x == x]
            if len(lost) >= 4 and sum(lost) < -1.0:
                h = len(lost) // 2
                early, late = sum(lost[:h]), sum(lost[h:])
                print(f'  new-old weight deficit: {sum(lost):.2f} total, '
                      f'{early:.2f} in the first half of growth, {late:.2f} '
                      f'in the second')
                if late < 2 * early:
                    diag.append('weight is lost mostly late in growth, after '
                                'many atoms are placed: dead ends from earlier '
                                'choices (look-ahead). Recoil growth or a '
                                'better growth order should help more than '
                                'extra trials')
                elif early < 2 * late:
                    diag.append('weight is lost mostly early in growth: the '
                                'first branches are poorly sampled; check the '
                                'trial counts and branch sampler of those '
                                'steps')
        if key in widom:
            w = widom[key]
            ess = float(w[6])
            print(f'  Widom ln<W_new> = {float(w[5]):.3f} over {w[4]} '
                  f'attempts, effective sample size {ess:.1f}')
            if ess < 30:
                diag.append('Widom estimate rests on fewer than 30 effective '
                            'insertions: treat ln<W> as a lower bound')
        for d in diag:
            print(f'  -> {d}')

    # Widom comparison between boxes for the same move and kind
    by = {}
    for key, w in widom.items():
        if key[3] == '0':
            by.setdefault((key[0], key[2]), {})[key[1]] = float(w[5])
    for (move, kind), boxes in by.items():
        if len(boxes) == 2:
            b = sorted(boxes)
            print(f'\nWidom {move} {kind}: ln<W> box {b[0]} - box {b[1]} = '
                  f'{boxes[b[0]] - boxes[b[1]]:.3f} '
                  f'(= beta*(mu_ex box {b[1]} - mu_ex box {b[0]}) when W uses '
                  f'the same ideal-chain reference)')


def report_nemtmc(sec, nwin):
    rows = sec.get('nemtmc', [])
    if not rows:
        return
    print('\n=== NeMTMC')
    for r in rows:
        counts = list(map(int, r[4:]))
        med = hist_quantile(counts, -20.0, 1.0, 0.5)
        lo5 = hist_quantile(counts, -20.0, 1.0, 0.05)
        hi95 = hist_quantile(counts, -20.0, 1.0, 0.95)
        print(f'  into box {r[0]} ({r[1]}): {r[2]} attempts, '
              f'{pct(int(r[3]), int(r[2])):.2f}% accepted, betaW median '
              f'{med:.1f}, 5-95% {lo5:.1f} to {hi95:.1f}')

    # Per-window dissipation. Exact only when one box is an ideal reservoir
    # (GCMC): then insertion (dest 0) traverses interval i upward in window i
    # and deletion (dest 1) traverses interval M-1-i downward.
    win = {}
    for r in sec.get('nemtmc_windows', []):
        win.setdefault((r[0], r[1]), []).append(
            (int(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])))
    kinds = {k for (_, k) in win}
    for kind in kinds:
        up, down = win.get(('0', kind)), win.get(('1', kind))
        if not up or not down or len(up) != len(down):
            continue
        M = len(up)
        lam = [w[1] for w in up] + [up[-1][2]]
        d = []
        for i in range(M):
            d.append(0.5 * (up[i][3] + down[M - 1 - i][3]))
        D = sum(d)
        print(f'\n  {kind}: dissipated work per window (valid for GCMC), '
              f'total {D:.2f} kT')
        print('    window  lam_lo   lam_hi   d (kT)')
        for i in range(M):
            print(f'    {i:5d}  {lam[i]:7.4f}  {lam[i + 1]:7.4f}  {d[i]:7.3f}')
        L = sum(math.sqrt(max(x, 1e-9)) for x in d)
        N = nwin or M
        print(f'    equal-length schedule with {N} windows: predicted '
              f'{L * L / N:.2f} kT (same RelaxingSteps per window)')
        if nwin:
            print('    LambdaVDW ' + ' '.join(
                f'{x:.4f}' for x in equal_length(lam, d, N)))


def equal_length(lam, d, N):
    """Place N+1 points at equal steps of l = sum sqrt(d_j) (linear within
    each interval)."""
    M = len(d)
    ell = [0.0]
    for j in range(M):
        ell.append(ell[-1] + math.sqrt(max(d[j], 1e-3 * max(d))))
    out = []
    for k in range(N + 1):
        t = ell[-1] * k / N
        j = 0
        while j < M - 1 and ell[j + 1] < t:
            j += 1
        f = (t - ell[j]) / (ell[j + 1] - ell[j]) if ell[j + 1] > ell[j] else 0
        out.append(lam[j] + f * (lam[j + 1] - lam[j]))
    out[0], out[-1] = 0.0 if lam[0] == 0.0 else lam[0], lam[-1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--all', action='store_true',
                    help='also report non-CBMC moves (displacement, rotation)')
    ap.add_argument('--schedule', type=int, default=0, metavar='N',
                    help='print an N-window NeMTMC schedule at equal '
                         'thermodynamic length (GCMC data)')
    a = ap.parse_args()
    sec = read_sections(a.file)
    if not sec:
        sys.exit(f'{a.file}: no instrumentation sections found')
    report_moves(sec, a.all)
    report_nemtmc(sec, a.schedule)


if __name__ == '__main__':
    main()
