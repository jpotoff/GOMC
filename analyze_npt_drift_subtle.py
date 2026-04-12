def read_ener(filename):
    data = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("ENER_0:"):
                    p = line.strip().split()
                    data.append({
                        'step': int(p[1]),
                        'recip': float(p[9]),
                        'vol': float(p[16])
                    })
    except Exception as e:
        pass
    return {d['step']: d for d in data}

pme = read_ener('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
ref = read_ener('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

if pme and ref:
    print(f"{'Step':>10} | {'PME_RECIP':>12} | {'REF_RECIP':>12} | {'dRECIP':>12} | {'PME_VOL':>12} | {'REF_VOL':>12} | {'dVOL':>10}")
    steps = sorted(list(set(pme.keys()).intersection(set(ref.keys()))))
    for s in steps[:20] + steps[-5:]:
        print(f"{s:>10} | {pme[s]['recip']:>12.0f} | {ref[s]['recip']:>12.0f} | {pme[s]['recip']-ref[s]['recip']:>12.0f} | {pme[s]['vol']:>12.0f} | {ref[s]['vol']:>12.0f} | {pme[s]['vol']-ref[s]['vol']:>10.0f}")
