import sys

def read_data(filename):
    data = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("ENER_0:"):
                    parts = line.strip().split()
                    step = int(parts[1])
                    total = float(parts[2])
                    total_elect = float(parts[7])
                    real = float(parts[8])
                    recip = float(parts[9])
                    self_e = float(parts[10])
                    data.append((step, total, total_elect, real, recip, self_e))
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return data

pme = read_data('/home/ai8111/PME/PME/NVT/OPC/out_NVT_equil.dat')
ref = read_data('/home/ai8111/PME/PME/NVT/OPC/reference/out_NVT_equil.dat')

if not pme:
    print("Could not read new PME data.")
    sys.exit(1)

print("="*120)
print(f"{'Step':>10s}  {'PME_TOTAL':>14s}  {'REF_TOTAL':>14s}  {'DIFF_TOTAL':>14s}  {'PME_RECIP':>14s}  {'REF_RECIP':>14s}  {'DIFF_RECIP':>14s}")
print("="*120)

min_len = min(len(pme), len(ref))
for i in range(min_len):
    sp, st, se, sr, srec, ss = pme[i]
    rp, rt, re, rr, rrec, rs = ref[i]
    if sp % 100000 == 0:
        print(f"{sp:>10d}  {st:>14.4f}  {rt:>14.4f}  {st-rt:>14.4f}  {srec:>14.4f}  {rrec:>14.4f}  {srec-rrec:>14.4f}")

# Check for stuck PME energy (no change between consecutive steps)
stuck_count = 0
for i in range(1, len(pme)):
    if abs(pme[i][4] - pme[i-1][4]) < 1.0:
        stuck_count += 1
print(f"\nNumber of consecutive steps where PME RECIP changes by < 1 K: {stuck_count} / {len(pme)-1}")

print("\n=== Average energies over last 5M steps (production) ===")
prod_start = 500  # step 5M = index 500
if len(pme) > prod_start:
    pme_prod = pme[prod_start:]
    ref_prod = ref[prod_start:len(pme)]

    pme_avg_total = sum(x[1] for x in pme_prod) / len(pme_prod)
    ref_avg_total = sum(x[1] for x in ref_prod) / len(ref_prod)
    pme_avg_recip = sum(x[4] for x in pme_prod) / len(pme_prod)
    ref_avg_recip = sum(x[4] for x in ref_prod) / len(ref_prod)

    print(f"PME avg TOTAL: {pme_avg_total:.4f},  REF avg TOTAL: {ref_avg_total:.4f},  DIFF: {pme_avg_total - ref_avg_total:.4f}")
    print(f"PME avg RECIP: {pme_avg_recip:.4f},  REF avg RECIP: {ref_avg_recip:.4f},  DIFF: {pme_avg_recip - ref_avg_recip:.4f}")

