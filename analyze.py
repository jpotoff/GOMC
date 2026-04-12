import sys

def read_data(filename):
    """Read STEP, TOTAL, TOTAL_ELECT, REAL, RECIP, SELF columns"""
    data = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            step = int(parts[0])
            total = float(parts[1])
            total_elect = float(parts[2])
            real = float(parts[3])
            recip = float(parts[4])
            self_e = float(parts[5])
            data.append((step, total, total_elect, real, recip, self_e))
    return data

pme = read_data('/home/ai8111/PME/GOMC/pme_data.txt')
ref = read_data('/home/ai8111/PME/GOMC/ref_data.txt')

print("="*120)
print(f"{'Step':>10s}  {'PME_TOTAL':>14s}  {'REF_TOTAL':>14s}  {'DIFF_TOTAL':>14s}  {'PME_RECIP':>14s}  {'REF_RECIP':>14s}  {'DIFF_RECIP':>14s}  {'PME_REAL':>14s}  {'REF_REAL':>14s}")
print("="*120)

# Print every 100k steps
for i in range(len(pme)):
    sp, st, se, sr, srec, ss = pme[i]
    rp, rt, re, rr, rrec, rs = ref[i]
    if sp % 100000 == 0:
        print(f"{sp:>10d}  {st:>14.4f}  {rt:>14.4f}  {st-rt:>14.4f}  {srec:>14.4f}  {rrec:>14.4f}  {srec-rrec:>14.4f}  {sr:>14.4f}  {rr:>14.4f}")

# Check for stuck PME energy (no change between consecutive steps)
print("\n=== Checking for stuck/frozen energy in PME ===")
stuck_count = 0
for i in range(1, len(pme)):
    if abs(pme[i][4] - pme[i-1][4]) < 1.0:  # RECIP barely changes
        stuck_count += 1
print(f"Number of consecutive steps where PME RECIP changes by < 1 K: {stuck_count} / {len(pme)-1}")

# Check for stuck energy runs
print("\n=== Longest runs of nearly identical PME RECIP ===")
run_start = 0
max_run = 0
max_run_start = 0
for i in range(1, len(pme)):
    if abs(pme[i][4] - pme[i-1][4]) < 1.0:
        run_len = i - run_start
        if run_len > max_run:
            max_run = run_len
            max_run_start = run_start
    else:
        run_start = i
print(f"Longest run of near-identical RECIP: {max_run} entries starting at step {pme[max_run_start][0]}")

# Last 500 steps comparison
print("\n=== Average energies over last 5M steps (production) ===")
prod_start = 500  # step 5M = index 500
pme_prod = pme[prod_start:]
ref_prod = ref[prod_start:]

pme_avg_total = sum(x[1] for x in pme_prod) / len(pme_prod)
ref_avg_total = sum(x[1] for x in ref_prod) / len(ref_prod)
pme_avg_recip = sum(x[4] for x in pme_prod) / len(pme_prod)
ref_avg_recip = sum(x[4] for x in ref_prod) / len(ref_prod)
pme_avg_real = sum(x[3] for x in pme_prod) / len(pme_prod)
ref_avg_real = sum(x[3] for x in ref_prod) / len(ref_prod)
pme_avg_elect = sum(x[2] for x in pme_prod) / len(pme_prod)
ref_avg_elect = sum(x[2] for x in ref_prod) / len(ref_prod)

print(f"PME avg TOTAL: {pme_avg_total:.4f},  REF avg TOTAL: {ref_avg_total:.4f},  DIFF: {pme_avg_total - ref_avg_total:.4f}")
print(f"PME avg ELECT: {pme_avg_elect:.4f},  REF avg ELECT: {ref_avg_elect:.4f},  DIFF: {pme_avg_elect - ref_avg_elect:.4f}")
print(f"PME avg REAL:  {pme_avg_real:.4f},  REF avg REAL:  {ref_avg_real:.4f},  DIFF: {pme_avg_real - ref_avg_real:.4f}")
print(f"PME avg RECIP: {pme_avg_recip:.4f},  REF avg RECIP: {ref_avg_recip:.4f},  DIFF: {pme_avg_recip - ref_avg_recip:.4f}")

