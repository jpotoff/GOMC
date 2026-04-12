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
                    })
    except Exception as e:
        pass
    return {d['step']: d['recip'] for d in data}

pme = read_ener('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
ref = read_ener('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

if pme and ref:
    print(f"{'Step':>10} | {'PME':>12} | {'REF':>12} | {'Diff':>12}")
    steps = sorted(list(set(pme.keys()).intersection(set(ref.keys()))))
    for s in steps[:50]:
        print(f"{s:>10} | {pme[s]:>12.0f} | {ref[s]:>12.0f} | {pme[s]-ref[s]:>12.0f}")

