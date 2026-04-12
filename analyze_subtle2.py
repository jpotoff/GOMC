import sys
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
    return data

pme = read_ener('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
ref = read_ener('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

limit = min(len(pme), len(ref))
if limit > 0:
    print(f"{'Idx':>5} | {'PME_RECIP':>12} | {'REF_RECIP':>12} | {'dRECIP':>12} | {'PME_VOL':>12} | {'REF_VOL':>12} | {'dVOL':>10}")
    for i in list(range(20)) + list(range(limit-5, limit)):
        if i < limit:
            pd = pme[i]
            rd = ref[i]
            print(f"{i:>5} | {pd['recip']:>12.0f} | {rd['recip']:>12.0f} | {pd['recip']-rd['recip']:>12.0f} | {pd['vol']:>12.0f} | {rd['vol']:>12.0f} | {pd['vol']-rd['vol']:>10.0f}")
