/*******************************************************************************
Intramolecular pair classes (Nonbond_1_3, Nonbond_1_4, Nonbond) are set by the
shortest bond path between two atoms. Each case builds a molecule with its
angles and dihedrals enumerated from the bonds, as a PSF lists them, and
compares the lists with an independent all-pairs shortest path (Floyd-
Warshall). Rings of four to seven atoms with two hydrogens per carbon cover
the cases where a dihedral's ends are closer by another path (four- and
five-membered rings); a chain checks that acyclic molecules keep the classes
the angle and dihedral based lists gave.
*******************************************************************************/

#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

#include "Geometry.h"
#include "MolSetup.h"

namespace {

typedef std::pair<uint, uint> Pair;

mol_setup::MolKind Molecule(uint nAtoms, const std::vector<Pair> &bonds) {
  mol_setup::MolKind m;
  for (uint i = 0; i < nAtoms; ++i)
    m.atoms.push_back(mol_setup::Atom("A", "RES", 1, "S", "T", 0.0, 1.0));
  std::vector<std::vector<uint>> adj(nAtoms);
  for (const Pair &b : bonds) {
    m.bonds.push_back(mol_setup::Bond(b.first, b.second));
    adj[b.first].push_back(b.second);
    adj[b.second].push_back(b.first);
  }
  // every angle a-b-c and dihedral a-b-c-d along the bonds, each once
  for (uint b = 0; b < nAtoms; ++b)
    for (uint a : adj[b])
      for (uint c : adj[b])
        if (a < c)
          m.angles.push_back(mol_setup::Angle(a, b, c));
  for (const Pair &bc : bonds) {
    uint b = bc.first, c = bc.second;
    for (uint a : adj[b])
      for (uint d : adj[c])
        if (a != c && d != b && a != d)
          m.dihedrals.push_back(mol_setup::Dihedral(a, b, c, d));
  }
  return m;
}

// Ring of n carbons, each with two hydrogens
mol_setup::MolKind Ring(uint n) {
  std::vector<Pair> bonds;
  for (uint i = 0; i < n; ++i)
    bonds.emplace_back(i, (i + 1) % n);
  for (uint i = 0; i < n; ++i) {
    bonds.emplace_back(i, n + 2 * i);
    bonds.emplace_back(i, n + 2 * i + 1);
  }
  return Molecule(3 * n, bonds);
}

std::set<Pair> AsSet(const Nonbond &nb) {
  std::set<Pair> s;
  for (uint i = 0; i < nb.count; ++i)
    s.emplace(nb.part1[i], nb.part2[i]);
  return s;
}

// Independent reference: pairs i < j at each shortest path length
void ReferenceClasses(const mol_setup::MolKind &m, std::set<Pair> &p13,
                      std::set<Pair> &p14, std::set<Pair> &p15) {
  const uint n = m.atoms.size(), inf = 1000;
  std::vector<std::vector<uint>> d(n, std::vector<uint>(n, inf));
  for (uint i = 0; i < n; ++i)
    d[i][i] = 0;
  for (const mol_setup::Bond &b : m.bonds)
    d[b.a0][b.a1] = d[b.a1][b.a0] = 1;
  for (uint k = 0; k < n; ++k)
    for (uint i = 0; i < n; ++i)
      for (uint j = 0; j < n; ++j)
        if (d[i][k] + d[k][j] < d[i][j])
          d[i][j] = d[i][k] + d[k][j];
  for (uint i = 0; i < n; ++i)
    for (uint j = i + 1; j < n; ++j) {
      if (d[i][j] == 2)
        p13.emplace(i, j);
      else if (d[i][j] == 3)
        p14.emplace(i, j);
      else if (d[i][j] >= 4)
        p15.emplace(i, j);
    }
}

void CheckClasses(const mol_setup::MolKind &m) {
  std::set<Pair> r13, r14, r15;
  ReferenceClasses(m, r13, r14, r15);
  Nonbond_1_3 nb13;
  Nonbond_1_4 nb14;
  Nonbond nb;
  nb13.Init(m);
  nb14.Init(m);
  nb.Init(m);
  EXPECT_EQ(AsSet(nb13), r13);
  EXPECT_EQ(AsSet(nb14), r14);
  EXPECT_EQ(AsSet(nb), r15);
  // no pair in two lists
  EXPECT_EQ(nb13.count + nb14.count + nb.count, r13.size() + r14.size() + r15.size());
  TopoDistance topo;
  topo.Init(m);
  for (const Pair &p : r14) {
    EXPECT_EQ(topo.Dist(p.first, p.second), 3u);
    EXPECT_EQ(TopoDistance::Between(m, p.second, p.first), 3u);
  }
}

} // namespace

TEST(TopoDistanceTest, Cyclobutane) {
  mol_setup::MolKind m = Ring(4);
  CheckClasses(m);
  // the bonded ends of the ring dihedral C0-C1-C2-C3 are not a 1-4 pair
  Nonbond_1_4 nb14;
  nb14.Init(m);
  EXPECT_EQ(AsSet(nb14).count(Pair(0, 3)), 0u);
}

TEST(TopoDistanceTest, Cyclopentane) {
  mol_setup::MolKind m = Ring(5);
  CheckClasses(m);
  // C0-C2 is joined by an angle and by the dihedral C0-C4-C3-C2: a 1-3 pair
  Nonbond_1_3 nb13;
  Nonbond_1_4 nb14;
  nb13.Init(m);
  nb14.Init(m);
  EXPECT_EQ(AsSet(nb13).count(Pair(0, 2)), 1u);
  EXPECT_EQ(AsSet(nb14).count(Pair(0, 2)), 0u);
}

TEST(TopoDistanceTest, CyclohexaneAndCycloheptane) {
  CheckClasses(Ring(6));
  CheckClasses(Ring(7));
}

// Acyclic molecule: the shortest path classes equal the angle and dihedral
// based ones (1-3 = angle ends, 1-4 = dihedral ends, 1-5+ = the rest)
TEST(TopoDistanceTest, BranchedChain) {
  // 2-methylhexane carbon skeleton with one hydrogen on each carbon
  std::vector<Pair> bonds = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {1, 6}};
  for (uint i = 0; i < 7; ++i)
    bonds.emplace_back(i, 7 + i);
  mol_setup::MolKind m = Molecule(14, bonds);
  CheckClasses(m);
  std::set<Pair> angEnds, dihEnds;
  for (const mol_setup::Angle &a : m.angles)
    angEnds.emplace(std::min(a.a0, a.a2), std::max(a.a0, a.a2));
  for (const mol_setup::Dihedral &d : m.dihedrals)
    dihEnds.emplace(std::min(d.a0, d.a3), std::max(d.a0, d.a3));
  Nonbond_1_3 nb13;
  Nonbond_1_4 nb14;
  nb13.Init(m);
  nb14.Init(m);
  EXPECT_EQ(AsSet(nb13), angEnds);
  EXPECT_EQ(AsSet(nb14), dihEnds);
}
