import sys
def read_data(filename):
    ener = {}
    stat = {}
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("ENER_0:"):
                    p = line.strip().split()
                    ener[int(p[1])] = float(p[9]) # recip
                elif line.startswith("STAT_0:"):
                    p = line.strip().split()
                    stat[int(p[1])] = float(p[2]) # vol
    except Exception as e:
        pass
    
    data = {}
    for step in ener:
        if step in stat:
            data[step] = {'recip': ener[step], 'vol': stat[step]}
    return data

pme = read_data('/home/ai8111/PME/PME/NPT/OPC/out_NPT_equil.dat')
ref = read_data('/home/ai8111/PME/PME/NPT/OPC/reference/out_NPT_equil.dat')

limit = min(len(pme), len(ref))
if limit > 0:
    print(f"{'Idx':>5} | {'Step':>10} | {'PME_RECIP':>12} | {'REF_RECIP':>12} | {'dRECIP':>12} | {'PME_VOL':>12} | {'REF_VOL':>12} | {'dVOL':>10}")
    steps = sorted(list(set(pme.keys()).intersection(set(ref.keys()))))
    for i, s in enumerate(steps[:20] + steps[-5:]):
        pd = pme[s]
        rd = ref[s]
        print(f"{i:>5} | {s:>10} | {pd['recip']:>12.0f} | {rd['recip']:>12.0f} | {pd['recip']-rd['recip']:>12.0f} | {pd['vol']:>12.0f} | {rd['vol']:>12.0f} | {pd['vol']-rd['vol']:>10.0f}")
