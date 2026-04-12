def read_ener(filename):
    data = []
    with open(filename) as f:
        for line in f:
            if line.startswith("ENER_0:"):
                p = line.strip().split()
                # STEP TOTAL INTRA(B) INTRA(NB) INTER(LJ) LRC TOTAL_ELECT REAL RECIP SELF CORR
                data.append({
                    'step': int(p[1]),
                    'total': float(p[2]),
                    'interLJ': float(p[5]),
                    'lrc': float(p[6]),
                    'elect': float(p[7]),
                    'real': float(p[8]),
                    'recip': float(p[9]),
                    'self': float(p[10]),
                    'corr': float(p[11]),
                })
    return data

pme = read_ener('/home/ai8111/PME/PME/NVT/OPC/out_NVT_equil.dat')
ref = read_ener('/home/ai8111/PME/PME/NVT/OPC/reference/out_NVT_equil.dat')

print(f"PME: {len(pme)} entries, last step={pme[-1]['step']}")
print(f"REF: {len(ref)} entries, last step={ref[-1]['step']}")

# Print first few and last few
print("\n=== First 10 entries ===")
print(f"{'Step':>10}  {'PME_RECIP':>14}  {'REF_RECIP':>14}  {'DIFF':>14}  {'PME_REAL':>14}  {'REF_REAL':>14}  {'DIFF_REAL':>14}  {'PME_CORR':>14}  {'REF_CORR':>14}  {'DIFF_CORR':>14}")
ml = min(len(pme), len(ref))
for i in range(min(10, ml)):
    p, r = pme[i], ref[i]
    print(f"{p['step']:>10}  {p['recip']:>14.4f}  {r['recip']:>14.4f}  {p['recip']-r['recip']:>14.4f}  "
          f"{p['real']:>14.4f}  {r['real']:>14.4f}  {p['real']-r['real']:>14.4f}  "
          f"{p['corr']:>14.4f}  {r['corr']:>14.4f}  {p['corr']-r['corr']:>14.4f}")

print("\n=== Last 10 entries ===")
for i in range(max(0, ml-10), ml):
    p, r = pme[i], ref[i]
    print(f"{p['step']:>10}  {p['recip']:>14.4f}  {r['recip']:>14.4f}  {p['recip']-r['recip']:>14.4f}  "
          f"{p['real']:>14.4f}  {r['real']:>14.4f}  {p['real']-r['real']:>14.4f}  "
          f"{p['corr']:>14.4f}  {r['corr']:>14.4f}  {p['corr']-r['corr']:>14.4f}")

# Averages over last 20% of production
n_prod = ml // 5
if n_prod > 10:
    pme_tail = pme[ml - n_prod:]
    ref_tail = ref[ml - n_prod:]
    
    keys = ['total', 'elect', 'real', 'recip', 'self', 'corr', 'interLJ', 'lrc']
    print("\n=== Averages over last 20% of data ===")
    print(f"{'Component':>12}  {'PME_avg':>14}  {'REF_avg':>14}  {'DIFF':>14}  {'%_diff':>10}")
    for k in keys:
        pa = sum(x[k] for x in pme_tail) / len(pme_tail)
        ra = sum(x[k] for x in ref_tail) / len(ref_tail)
        pct = abs(pa - ra) / max(abs(ra), 1e-30) * 100
        print(f"{k:>12}  {pa:>14.4f}  {ra:>14.4f}  {pa-ra:>14.4f}  {pct:>10.4f}%")

# Check stuck recip
stuck = 0
for i in range(1, len(pme)):
    if abs(pme[i]['recip'] - pme[i-1]['recip']) < 1.0:
        stuck += 1
print(f"\nStuck PME RECIP (<1 K change): {stuck}/{len(pme)-1}")

# Check step 0 initial energies in detail
if pme and ref:
    p0, r0 = pme[0], ref[0]
    print("\n=== Step 0 detailed comparison ===")
    for k in ['total', 'elect', 'real', 'recip', 'self', 'corr', 'interLJ', 'lrc']:
        print(f"  {k:>12}: PME={p0[k]:>18.8f}  REF={r0[k]:>18.8f}  diff={p0[k]-r0[k]:>14.8f}")

