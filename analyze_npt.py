def read_ener(filename):
    data = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("ENER_0:"):
                    p = line.strip().split()
                    data.append({
                        'step': int(p[1]),
                        'total': float(p[2]),
                        'elect': float(p[7]),
                        'real': float(p[8]),
                        'recip': float(p[9]),
                    })
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return data

def read_box(filename):
    data = []
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("STAT_0:"):
                    p = line.strip().split()
                    data.append({
                        'step': int(p[1]),
                        'volume': float(p[2]),
                        'density': float(p[4]),
                    })
    except Exception as e:
        pass
    return data

pme = read_ener('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
ref = read_ener('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

pb = read_box('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
rb = read_box('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

if not pme or not ref:
    print("Missing energy data")
    
ml = min(len(pme), len(ref))
mlb = min(len(pb), len(rb))

print("\n=== NPT Equilibration Trend ===")
print(f"{'Step':>10} | {'PME_RECIP':>12} {'REF_RECIP':>12} {'dRECIP':>12} | {'PME_REAL':>12} {'REF_REAL':>12} | {'PME_VOL':>10} {'REF_VOL':>10} {'dVOL':>10}")

for i in range(min(ml, mlb)):
    if i % max(1, ml // 10) == 0 or i == ml - 1:
        sp = pme[i]['step']
        pr = pme[i]['recip']
        rr = ref[i]['recip']
        preal = pme[i]['real']
        rreal = ref[i]['real']
        
        pv = pb[i]['volume']
        rv = rb[i]['volume']
        
        print(f"{sp:>10} | {pr:>12.0f} {rr:>12.0f} {pr-rr:>12.0f} | {preal:>12.0f} {rreal:>12.0f} | {pv:>10.0f} {rv:>10.0f} {pv-rv:>10.0f}")

