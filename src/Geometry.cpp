/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "Geometry.h"

#include <algorithm>
#include <vector>

#include "FFSetup.h"
#include "MolSetup.h"

namespace {
// returns where p1 and p2 are found as a pair in bList, index+count if backward
int findPair(uint p1, uint p2, const BondList &bList) {
  for (uint i = 0; i < bList.count; ++i) {
    if (bList.part1[i] == p1 && bList.part2[i] == p2)
      return i;
    else if (bList.part1[i] == p2 && bList.part2[i] == p1)
      return i + bList.count;
  }
  return -1;
}
} // namespace

Nonbond::Nonbond() : part1(NULL), part2(NULL), count(0) {}

Nonbond::~Nonbond() {
  delete[] part1;
  delete[] part2;
}

namespace {
// Breadth-first search from atom a over the bonds, up to three bonds.
// dist[x] is the shortest path from a, or TopoDistance::kFar if beyond.
void NearAtoms(const std::vector<std::vector<uint>> &adj, uint a,
               std::vector<uint> &dist, std::vector<uint> &visited) {
  for (uint v : visited)
    dist[v] = TopoDistance::kFar;
  visited.clear();
  dist[a] = 0;
  visited.push_back(a);
  for (size_t head = 0; head < visited.size(); ++head) {
    uint u = visited[head];
    if (dist[u] + 1 >= TopoDistance::kFar)
      continue;
    for (uint w : adj[u]) {
      if (dist[w] == TopoDistance::kFar) {
        dist[w] = dist[u] + 1;
        visited.push_back(w);
      }
    }
  }
}

std::vector<std::vector<uint>> Adjacency(const mol_setup::MolKind &molData) {
  std::vector<std::vector<uint>> adj(molData.atoms.size());
  for (const mol_setup::Bond &b : molData.bonds) {
    adj[b.a0].push_back(b.a1);
    adj[b.a1].push_back(b.a0);
  }
  return adj;
}

// All pairs i < j whose shortest path satisfies keep(distance), ordered by i
// and then j, as the angle/dihedral based lists were
template <class Keep>
void PairsByDistance(const mol_setup::MolKind &molData, Keep keep,
                     std::vector<uint> &part1Vec, std::vector<uint> &part2Vec) {
  const uint numAtoms = molData.atoms.size();
  std::vector<std::vector<uint>> adj = Adjacency(molData);
  std::vector<uint> dist(numAtoms, TopoDistance::kFar), visited;
  for (uint i = 0; i < numAtoms; ++i) {
    NearAtoms(adj, i, dist, visited);
    for (uint j = i + 1; j < numAtoms; ++j) {
      if (keep(dist[j])) {
        part1Vec.push_back(i);
        part2Vec.push_back(j);
      }
    }
  }
}

void StorePairs(const std::vector<uint> &part1Vec,
                const std::vector<uint> &part2Vec, uint *&part1, uint *&part2,
                uint &count) {
  count = part1Vec.size();
  part1 = new uint[count];
  part2 = new uint[count];
  std::copy(part1Vec.begin(), part1Vec.end(), part1);
  std::copy(part2Vec.begin(), part2Vec.end(), part2);
}
} // namespace

void TopoDistance::Init(const mol_setup::MolKind &molData) {
  const uint numAtoms = molData.atoms.size();
  std::vector<std::vector<uint>> adj = Adjacency(molData);
  std::vector<uint> dist(numAtoms, kFar), visited;
  near.assign(numAtoms, {});
  for (uint a = 0; a < numAtoms; ++a) {
    NearAtoms(adj, a, dist, visited);
    for (uint v : visited)
      near[a].emplace_back(v, dist[v]);
  }
}

uint TopoDistance::Dist(uint a, uint b) const {
  for (const std::pair<uint, uint> &p : near[a])
    if (p.first == b)
      return p.second;
  return kFar;
}

uint TopoDistance::Between(const mol_setup::MolKind &molData, uint a, uint b) {
  std::vector<std::vector<uint>> adj = Adjacency(molData);
  std::vector<uint> dist(molData.atoms.size(), kFar), visited;
  NearAtoms(adj, a, dist, visited);
  return dist[b];
}

// 1-5 and more: pairs four or more bonds apart (or not connected)
void Nonbond::Init(const mol_setup::MolKind &molData) {
  // no nonbonded atoms
  if (molData.atoms.size() < 5) {
    return;
  }
  std::vector<uint> part1Vec, part2Vec;
  PairsByDistance(
      molData, [](uint d) { return d >= TopoDistance::kFar; }, part1Vec,
      part2Vec);
  StorePairs(part1Vec, part2Vec, part1, part2, count);
}

// 1-4: pairs whose shortest path is three bonds
void Nonbond_1_4::Init(const mol_setup::MolKind &molData) {
  // no nonbonded atoms
  if (molData.atoms.size() < 4) {
    return;
  }
  std::vector<uint> part1Vec, part2Vec;
  PairsByDistance(
      molData, [](uint d) { return d == 3; }, part1Vec, part2Vec);
  StorePairs(part1Vec, part2Vec, part1, part2, count);
}

// 1-3: pairs whose shortest path is two bonds
void Nonbond_1_3::Init(const mol_setup::MolKind &molData) {
  // no nonbonded atoms
  if (molData.atoms.size() < 3) {
    return;
  }
  std::vector<uint> part1Vec, part2Vec;
  PairsByDistance(
      molData, [](uint d) { return d == 2; }, part1Vec, part2Vec);
  StorePairs(part1Vec, part2Vec, part1, part2, count);
}

void EwaldNonbond::Init(const mol_setup::MolKind &molData) {
  unsigned int numAtoms = molData.atoms.size();
  std::vector<unsigned int> part1Vec;
  std::vector<unsigned int> part2Vec;
  std::vector<char> nonBondedAtoms(numAtoms);
  // find all possible pairs
  for (unsigned int i = 0; i < numAtoms; ++i) {
    // starting at i+1 to ignore double counting

    for (unsigned int j = i + 1; j < numAtoms; ++j) {
      part1Vec.push_back(i);
      part2Vec.push_back(j);
    }
  }
  count = part1Vec.size();
  part1 = new uint[count];
  part2 = new uint[count];
  std::copy(part1Vec.begin(), part1Vec.end(), part1);
  std::copy(part2Vec.begin(), part2Vec.end(), part2);
}

void BondList::Init(const std::vector<mol_setup::Bond> &bonds) {
  count = bonds.size();
  part1 = new uint[count];
  part2 = new uint[count];
  kinds = new uint[count];
  for (uint i = 0; i < count; ++i) {
    part1[i] = bonds[i].a0;
    part2[i] = bonds[i].a1;
    kinds[i] = bonds[i].kind;
  }
}

bool BondList::IsBonded(const uint &ai, const uint &aj) {
  for (uint i = 0; i < count; ++i) {
    if (part1[i] == ai && part2[i] == aj)
      return true;
    else if (part1[i] == aj && part2[i] == ai)
      return true;
  }
  return false;
}

BondList::BondList() : part1(NULL), part2(NULL), kinds(NULL), count(0) {}

BondList::~BondList() {
  delete[] part1;
  delete[] part2;
  delete[] kinds;
}

GeomFeature::GeomFeature(uint atomsPer)
    : bondsPer(atomsPer - 1), bondIndices(NULL), kinds(NULL), count(0) {}

GeomFeature::~GeomFeature() {
  delete[] bondIndices;
  delete[] kinds;
}

void GeomFeature::Init(const std::vector<mol_setup::Angle> &angles,
                       const BondList &bList) {
  count = angles.size();
  if (count == 0)
    return;
  // find corresponding bond indices
  kinds = new uint[count];
  bondIndices = new uint[count * bondsPer];

  int bondCounter = 0;
  for (uint i = 0; i < angles.size(); ++i) {
    bondIndices[bondCounter] = findPair(angles[i].a0, angles[i].a1, bList);
    ++bondCounter;
    bondIndices[bondCounter] = findPair(angles[i].a1, angles[i].a2, bList);
    ++bondCounter;
    kinds[i] = angles[i].kind;
  }
}

void GeomFeature::Init(const std::vector<mol_setup::Dihedral> &dihs,
                       const BondList &bList) {
  count = dihs.size();
  if (count == 0)
    return;
  // find corresponding bond indices
  kinds = new uint[count];
  bondIndices = new uint[count * bondsPer];

  int bondCounter = 0;
  for (uint i = 0; i < dihs.size(); ++i) {
    bondIndices[bondCounter] = findPair(dihs[i].a0, dihs[i].a1, bList);
    ++bondCounter;
    bondIndices[bondCounter] = findPair(dihs[i].a1, dihs[i].a2, bList);
    ++bondCounter;
    bondIndices[bondCounter] = findPair(dihs[i].a2, dihs[i].a3, bList);
    ++bondCounter;
    kinds[i] = dihs[i].kind;
  }
}

void GeomFeature::Init(const std::vector<mol_setup::Improper> &imps,
                       const BondList &bList) {
  count = imps.size();
  if (count == 0)
    return;
  // find corresponding bond indices
  kinds = new uint[count];
  bondIndices = new uint[count * bondsPer];

  int bondCounter = 0;
  for (uint i = 0; i < imps.size(); ++i) {
    bondIndices[bondCounter] = findPair(imps[i].a0, imps[i].a1, bList);
    ++bondCounter;
    bondIndices[bondCounter] = findPair(imps[i].a1, imps[i].a2, bList);
    ++bondCounter;
    bondIndices[bondCounter] = findPair(imps[i].a2, imps[i].a3, bList);
    ++bondCounter;
    kinds[i] = imps[i].kind;
  }
}

void SortedNonbond::Init(const Nonbond &nb, const uint numAtoms) {
  std::vector<uint> partnerVec;
  uint oldSize = 0;
  subdiv.Init(numAtoms);
  for (uint i = 0; i < numAtoms; ++i) {
    oldSize = partnerVec.size();
    for (uint j = 0; j < nb.count; ++j) {
      if (nb.part1[j] == i)
        partnerVec.push_back(nb.part2[j]);
      else if (nb.part2[j] == i)
        partnerVec.push_back(nb.part1[j]);
    }
    subdiv.Set(i, oldSize, partnerVec.size() - oldSize);
    // maybe sorting will improve access patterns?
    std::sort(partnerVec.begin() + oldSize, partnerVec.end());
  }
  partners = new uint[partnerVec.size()];
  std::copy(partnerVec.begin(), partnerVec.end(), partners);
}
