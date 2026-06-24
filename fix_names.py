import os

# Rename files
os.rename('/home/ai8111/GOMC/GOMC/test/src/TestEwaldCachedMoves.cpp', '/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp')
os.rename('/home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwaldCached.cpp', '/home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwald.cpp')

# Update FileList.cmake
with open('/home/ai8111/GOMC/GOMC/test/FileList.cmake', 'r') as f:
    content = f.read()

content = content.replace('TestEwaldCachedMoves.cpp', 'TestEwaldMoves.cpp')
content = content.replace('TestMultiBoxEwaldCached.cpp', 'TestMultiBoxEwald.cpp')

with open('/home/ai8111/GOMC/GOMC/test/FileList.cmake', 'w') as f:
    f.write(content)

# Update TestEwaldMoves.cpp
with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'r') as f:
    content = f.read()

content = content.replace('EwaldCachedMovesTest', 'EwaldMovesTest')
content = content.replace('EwaldCached', 'Ewald')
content = content.replace('Ewald.h.h', 'Ewald.h')
content = content.replace('"CachedFourier True\\n"', '"CachedFourier False\\n"')
content = content.replace('exgMolCache', 'backupMolCache') # wait, exgMolCache is in Ewald too, but let's check. Ewald base has exgMolCache

with open('/home/ai8111/GOMC/GOMC/test/src/TestEwaldMoves.cpp', 'w') as f:
    f.write(content)

# Update TestMultiBoxEwald.cpp
with open('/home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwald.cpp', 'r') as f:
    content = f.read()

content = content.replace('EwaldCachedMultiBoxTest', 'EwaldMultiBoxTest')
content = content.replace('EwaldCached', 'Ewald')
content = content.replace('Ewald.h.h', 'Ewald.h')

with open('/home/ai8111/GOMC/GOMC/test/src/TestMultiBoxEwald.cpp', 'w') as f:
    f.write(content)

