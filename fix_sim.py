with open('/home/ai8111/GOMC/GOMC/src/Simulation.cpp', 'r') as f:
    content = f.read()

content = content.replace('ulong Simulation::GetRunSteps() { return totalSteps - startStep; }\n#endif', 'ulong Simulation::GetRunSteps() { return totalSteps - startStep; }\nEwald *Simulation::GetEwald() { return system->GetEwald(); }\nCalculateEnergy &Simulation::GetCalcEnergy() { return system->calcEnergy; }\n#endif')

with open('/home/ai8111/GOMC/GOMC/src/Simulation.cpp', 'w') as f:
    f.write(content)
