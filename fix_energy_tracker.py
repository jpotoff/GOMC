import re

with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'r') as f:
    content = f.read()

# Fix simple dE cases
content = re.sub(
    r'// In GOMC with PME, the energy tracker is now directly updated to the\n\s*// exact reciprocal energy within UpdateRecip\.\n\s*double expectedEnergy = sim\.GetSystemEnergy\(\)\.boxEnergy\[box\]\.recip;',
    'sim.GetSystemEnergy().boxEnergy[box].recip += dE;\n  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;',
    content
)

# Fix SwapMoveConsistency which has dE_del and dE_ins
content = re.sub(
    r'// Net expected tracking\n\s*double expectedEnergy = sim\.GetSystemEnergy\(\)\.boxEnergy\[box\]\.recip;',
    'sim.GetSystemEnergy().boxEnergy[box].recip += (dE_del + dE_ins);\n  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;',
    content
)

# Fix CombinedVolumeAndDisplacementMoveConsistency
content = re.sub(
    r'// Accept Displacement\n\s*pme->UpdateRecip\(box\);\n\s*double expectedEnergy = sim\.GetSystemEnergy\(\)\.boxEnergy\[box\]\.recip;',
    '// Accept Displacement\n  pme->UpdateRecip(box);\n  sim.GetSystemEnergy().boxEnergy[box].recip += dE;\n  double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;',
    content
)

# Fix RejectedDisplacementConsistency
content = re.sub(
    r'// In GOMC with PME, the energy tracker is untouched if move is rejected\.\n\s*double expectedEnergy = sim\.GetSystemEnergy\(\)\.boxEnergy\[box\]\.recip;',
    'double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;',
    content
)

# Fix RejectedVolumeDoesNotCorruptPerMoveEnergy loop
content = re.sub(
    r'double expectedEnergy = sim\.GetSystemEnergy\(\)\.boxEnergy\[box\]\.recip;',
    'sim.GetSystemEnergy().boxEnergy[box].recip += dE;\n      double expectedEnergy = sim.GetSystemEnergy().boxEnergy[box].recip;',
    content
)

with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'w') as f:
    f.write(content)
