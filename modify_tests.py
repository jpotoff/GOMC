import os

def modify_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

replacements_main = [
    ('EwaldPMEMovesTest', 'EwaldCachedMovesTest'),
    ('EwaldPME', 'EwaldCached'),
    ('EwaldPME.h', 'EwaldCached.h'),
    ('"ElectrostaticMethod PME\\n"', '"Ewald True\\n");\n    fprintf(f, "CachedFourier True\\n"'),
    ('fprintf(f, "PMESplineOrder 4\\n");\n', ''),
    ('fprintf(f, "PMEGridSpacing 1.5\\n");\n', ''),
    ('fprintf(f, "PMEGridSpacing 1.5 4.0\\n");\n', ''),
    ('fprintf(f, "PMERefreshFreq 100\\n");\n', ''),
    ('gomc_pme_repro', 'gomc_ewald_repro')
]

modify_file('/home/ai8111/GOMC/GOMC/test/src/TestEwaldCachedMoves.cpp', replacements_main)

# Copy and modify TestMultiBox
os.system("cp /home/ai8111/PME/revert/GOMC/test/src/TestMultiBoxPME.cpp /home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwaldCached.cpp")
replacements_multibox = [
    ('EwaldPMEMultiBoxTest', 'EwaldCachedMultiBoxTest'),
    ('EwaldPME', 'EwaldCached'),
    ('EwaldPME.h', 'EwaldCached.h')
]
modify_file('/home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwaldCached.cpp', replacements_multibox)

# Update FileList.cmake
with open('/home/ai8111/GOMC/GOMC/test/FileList.cmake', 'r') as f:
    content = f.read()

if 'test/src/TestEwaldCachedMoves.cpp' not in content:
    content = content.replace('test/src/CheckpointTest.cpp', 'test/src/CheckpointTest.cpp\n    test/src/TestEwaldCachedMoves.cpp\n    test/src/TestMultiBoxEwaldCached.cpp')
    with open('/home/ai8111/GOMC/GOMC/test/FileList.cmake', 'w') as f:
        f.write(content)

