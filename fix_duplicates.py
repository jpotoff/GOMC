import re

with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'r') as f:
    content = f.read()

# Fix double += dE
content = content.replace(
    'sim.GetSystemEnergy().boxEnergy[box].recip += dE;\n  sim.GetSystemEnergy().boxEnergy[box].recip += dE;',
    'sim.GetSystemEnergy().boxEnergy[box].recip += dE;'
)

# Fix SwapMoveConsistency which has += (dE_del + dE_ins) then += dE
content = content.replace(
    'sim.GetSystemEnergy().boxEnergy[box].recip += (dE_del + dE_ins);\n  sim.GetSystemEnergy().boxEnergy[box].recip += dE;',
    'sim.GetSystemEnergy().boxEnergy[box].recip += (dE_del + dE_ins);'
)

# Fix CombinedVolumeAndDisplacementMoveConsistency where it might be messed up
# Wait, let's see how it looks
with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'w') as f:
    f.write(content)

