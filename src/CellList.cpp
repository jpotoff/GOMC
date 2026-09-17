/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "CellList.h"

#include <algorithm>

#include "BoxDimensions.h"
#include "BoxDimensionsNonOrth.h"
#include "MoleculeLookup.h"
#include "Molecules.h"
#include "XYZArray.h"

const int CellList::END_CELL;

CellList::CellList(const Molecules &mols, BoxDimensions &dims) : mols(&mols) {
  dimensions = &dims;
  isBuilt = false;
  for (uint b = 0; b < BOX_TOTAL; b++) {
    edgeCells[b][0] = edgeCells[b][1] = edgeCells[b][2] = 0;
    stencil[b][0] = stencil[b][1] = stencil[b][2] = 1;
  }
}

CellList::CellList(const CellList &other) : mols(other.mols) {
  dimensions = other.dimensions;
  isBuilt = true;
  for (uint b = 0; b < BOX_TOTAL; b++) {
    edgeCells[b][0] = other.edgeCells[b][0];
    edgeCells[b][1] = other.edgeCells[b][1];
    edgeCells[b][2] = other.edgeCells[b][2];
    stencil[b][0] = other.stencil[b][0];
    stencil[b][1] = other.stencil[b][1];
    stencil[b][2] = other.stencil[b][2];
  }

  for (uint b = 0; b < BOX_TOTAL; b++) {
    RebuildNeighbors(b);
  }

  list.resize(other.list.size());

  for (size_t i = 0; i < other.list.size(); i++) {
    list[i] = other.list[i];
  }

  for (uint b = 0; b < BOX_TOTAL; b++) {
    for (size_t i = 0; i < other.neighbors[b].size(); i++) {
      neighbors[b][i] = other.neighbors[b][i];
    }
  }

  for (uint b = 0; b < BOX_TOTAL; b++) {
    for (size_t i = 0; i < other.head[b].size(); i++) {
      head[b][i] = other.head[b][i];
    }
  }
  // neighbors(other.neighbors);
  // head(other.head);
}

void CellList::SetCutoff() {
  for (uint b = 0; b < BOX_TOTAL; b++) {
    cutoff[b] = dimensions->rCut[b];
  }
}

bool CellList::IsExhaustive() const {
  std::vector<int> particles(list);
  for (int b = 0; b < BOX_TOTAL; ++b) {
    particles.insert(particles.end(), head[b].begin(), head[b].end());
  }
  particles.erase(std::remove(particles.begin(), particles.end(), -1),
                  particles.end());
  std::sort(particles.begin(), particles.end());
  for (int i = 0; i < (int)particles.size(); ++i) {
    if (i != particles[i])
      return false;
  }
  return true;
}

void CellList::RemoveMol(const int molIndex, const int box,
                         const XYZArray &pos) {
  // For each atom in molecule
  int p = mols->MolStart(molIndex);
  int end = mols->MolEnd(molIndex);
  while (p != end) {
    int cell = PositionToCell(pos[p], box);
    int at = head[box][cell];

    // If particle we're looking for is at head of list assign its
    // pointed at index (should be -1) to head.... this is the case
    // for removing the head molecule in the cell, which is often when
    // we're removing the last molecule/particle from a particular cell.
    //
    // If particle isn't at the head of the list, traverse links to find it,
    // relinking once found.
    if (at == p) {
      head[box][cell] = list[p];
    } else {
      while (at != END_CELL) {
        if (list[at] == p) {
          list[at] = list[p];
          break;
        }
        at = list[at];
      }
    }
    ++p;
  }
}

void CellList::AddMol(const int molIndex, const int box, const XYZArray &pos) {
  // For each atom in molecule
  int p = mols->MolStart(molIndex);
  int end = mols->MolEnd(molIndex);

  // Note: GridAll assigns everything to END_CELL
  // so list should point to that
  // if this is the first particle in a particular cell.
  while (p != end) {
    int cell = PositionToCell(pos[p], box);
#ifndef NDEBUG
    if (cell >= static_cast<int>(head[box].size())) {
      std::cout << "CellList.cpp:129: box " << box
                << ", pos out of cell: " << pos[p] << std::endl;
      std::cout << "AxisDimensions: " << dimensions->GetAxis(box) << std::endl;
    }
#endif
    // Make the current head index the index the new head points at.
    list[p] = head[box][cell];
    // Assign the new head as our particle index
    head[box][cell] = p;
    ++p;
  }
}


//
// Pick the cell grid and the matching stencil radius for one box.
//
// A stencil of +-R cells around a particle's own cell is only correct if R
// cells span the cutoff. The classic choice -- cell edge = cutoff, R = 1, 27
// cells -- breaks down when the box is narrow: `floor(side/cutoff)` is clamped
// to a minimum of 3 cells per side, and a +-1 stencil over a 3x3x3 grid wraps
// onto *every* cell. The cell list then accelerates nothing and the pair loop
// runs all-pairs. A GEMC liquid box hits this squarely: 32.8 A with
// rCut = max(rCut, rCutCoulomb) = 12 A gives 3x3x3 cells covering 100% of the
// box, of which only ~20% of the pairs tested are actually inside the cutoff.
//
// Finer cells with a wider stencil cover less: (2R+1)^3 cells of edge L/n cover
// ((2R+1)/n)^3 of the volume, which falls as n grows. So search n for the
// smallest coverage, subject to two constraints:
//
//   n >= 2R+1   -- otherwise the stencil wraps onto itself and a pair would be
//                  visited (and counted) more than once.
//   n^3 * (2R+1)^3 <= kMaxNeighborEntries -- the neighbour lists are rebuilt on
//                  every volume move, so they have to stay small.
//
// A cubic stencil cannot do better than (2*rCut)^3 / (4/3 pi rCut^3) = 1.9x the
// sphere, so this buys roughly 2x fewer distance tests, not the 5x that perfect
// spherical culling would give.
//
static void ChooseGrid(const XYZ &sides, double cutoff, int *eCells,
                       XYZ &cellSize, int *stencil) {
  // Neighbour-list entries we are willing to rebuild per volume move (~16 MB).
  const long kMaxNeighborEntries = 4000000L;
  const double side[3] = {sides.x, sides.y, sides.z};

  int bestN[3] = {0, 0, 0}, bestR[3] = {0, 0, 0};
  double bestCoverage = 2.0;

  // n is the number of cells along the shortest axis; the other axes get a
  // proportional count so cells stay roughly cubic.
  const double minSide = std::min(side[0], std::min(side[1], side[2]));
  for (int n = 3; n <= 64; ++n) {
    int nc[3], r[3];
    double coverage = 1.0;
    long entries = 1, cells = 1;
    bool ok = true;
    for (int d = 0; d < 3; ++d) {
      nc[d] = std::max((int)floor(n * side[d] / minSide), 3);
      const double cs = side[d] / nc[d];
      r[d] = (int)std::ceil(cutoff / cs - 1e-12);
      if (2 * r[d] + 1 > nc[d]) { // stencil would wrap onto itself
        ok = false;
        break;
      }
      coverage *= (double)(2 * r[d] + 1) / nc[d];
      cells *= nc[d];
      entries *= (2 * r[d] + 1);
    }
    if (!ok || cells * entries > kMaxNeighborEntries)
      continue;
    if (coverage < bestCoverage) {
      bestCoverage = coverage;
      for (int d = 0; d < 3; ++d) {
        bestN[d] = nc[d];
        bestR[d] = r[d];
      }
    }
  }

  if (bestCoverage > 1.5) {
    // Nothing valid -- fall back to the original cell-edge-equals-cutoff grid
    // with a +-1 stencil, which is always correct even when it degenerates.
    for (int d = 0; d < 3; ++d) {
      bestN[d] = std::max((int)floor(side[d] / cutoff), 3);
      bestR[d] = 1;
    }
  }

  for (int d = 0; d < 3; ++d) {
    eCells[d] = bestN[d];
    stencil[d] = bestR[d];
  }
  cellSize.x = side[0] / eCells[0];
  cellSize.y = side[1] / eCells[1];
  cellSize.z = side[2] / eCells[2];
}

// Resize all boxes to match current axes
void CellList::ResizeGrid(const BoxDimensions &dims) {
  for (uint b = 0; b < BOX_TOTAL; ++b) {
    XYZ sides = dims.axis[b];
    bool rebuild = false;
    int *eCells = edgeCells[b];
    int oldCells[3] = {eCells[0], eCells[1], eCells[2]};
    int oldStencil[3] = {stencil[b][0], stencil[b][1], stencil[b][2]};
    ChooseGrid(sides, cutoff[b], eCells, cellSize[b], stencil[b]);
    rebuild |= !isBuilt;
    for (int d = 0; d < 3; ++d)
      rebuild |= (oldCells[d] != eCells[d]) || (oldStencil[d] != stencil[b][d]);

    if (rebuild) {
      RebuildNeighbors(b);
    }
  }
  isBuilt = true;
}

// Resize one boxes to match current axes
void CellList::ResizeGridBox(const BoxDimensions &dims, const uint b) {
  XYZ sides = dims.axis[b];
  bool rebuild = false;
  int *eCells = edgeCells[b];
  int oldCells[3] = {eCells[0], eCells[1], eCells[2]};
  int oldStencil[3] = {stencil[b][0], stencil[b][1], stencil[b][2]};
  ChooseGrid(sides, cutoff[b], eCells, cellSize[b], stencil[b]);
  rebuild |= !isBuilt;
  for (int d = 0; d < 3; ++d)
    rebuild |= (oldCells[d] != eCells[d]) || (oldStencil[d] != stencil[b][d]);

  if (rebuild) {
    RebuildNeighbors(b);
  }
  isBuilt = true;
}

void CellList::RebuildNeighbors(int b) {
  int *eCells = edgeCells[b];
  int nCells = eCells[0] * eCells[1] * eCells[2];
  head[b].resize(nCells);
  neighbors[b].resize(nCells);
  for (int i = 0; i < nCells; ++i) {
    neighbors[b][i].clear();
  }

  for (int x = 0; x < eCells[0]; ++x) {
    for (int y = 0; y < eCells[1]; ++y) {
      for (int z = 0; z < eCells[2]; ++z) {
        int cell = x * eCells[2] * eCells[1] + y * eCells[2] + z;
        for (int dx = -stencil[b][0]; dx <= stencil[b][0]; ++dx) {
          for (int dy = -stencil[b][1]; dy <= stencil[b][1]; ++dy) {
            for (int dz = -stencil[b][2]; dz <= stencil[b][2]; ++dz) {
              // Cache adjacent cells, wrapping if needed
              neighbors[b][cell].push_back(
                  ((x + dx + eCells[0]) % eCells[0]) * eCells[2] * eCells[1] +
                  ((y + dy + eCells[1]) % eCells[1]) * eCells[2] +
                  ((z + dz + eCells[2]) % eCells[2]));
            }
          }
        }
      }
    }
  }
}

void CellList::GridAll(BoxDimensions &dims, const XYZArray &pos,
                       const MoleculeLookup &lookup) {
  dimensions = &dims;
  list.resize(pos.Count());
  ResizeGrid(dims);
  for (int b = 0; b < BOX_TOTAL; ++b) {
    head[b].assign(edgeCells[b][0] * edgeCells[b][1] * edgeCells[b][2],
                   END_CELL);
    MoleculeLookup::box_iterator it = lookup.BoxBegin(b),
                                 end = lookup.BoxEnd(b);

    // For each molecule per box
    while (it != end) {
      AddMol(*it, b, pos);
      ++it;
    }
  }
}

void CellList::GridBox(BoxDimensions &dims, const XYZArray &pos,
                       const MoleculeLookup &lookup, const uint b) {
  dimensions = &dims;
  list.resize(pos.Count());
  ResizeGridBox(dims, b);
  head[b].assign(edgeCells[b][0] * edgeCells[b][1] * edgeCells[b][2], END_CELL);
  MoleculeLookup::box_iterator it = lookup.BoxBegin(b), end = lookup.BoxEnd(b);

  // For each molecule per box
  while (it != end) {
    AddMol(*it, b, pos);
    ++it;
  }
}

CellList::Pairs CellList::EnumeratePairs(int box) const {
  return CellList::Pairs(*this, box);
}

void CellList::GetCellListNeighbor(uint box, int coordinateSize,
                                   std::vector<int> &cellVector,
                                   std::vector<int> &cellStartIndex,
                                   std::vector<int> &mapParticleToCell) const {
  cellVector.resize(coordinateSize);
  cellStartIndex.resize(head[box].size());
  mapParticleToCell.resize(coordinateSize);
  int vector_index = 0;
  for (size_t cell = 0; cell < head[box].size(); cell++) {
    cellStartIndex[cell] = vector_index;
    int particleIndex = head[box][cell];
    while (particleIndex != END_CELL) {
      cellVector[vector_index] = particleIndex;
      mapParticleToCell[particleIndex] = cell;
      vector_index++;
      particleIndex = list[particleIndex];
    }
    // we are going to sort particles in each cell for better memory access
    std::sort(cellVector.begin() + cellStartIndex[cell],
              cellVector.begin() + vector_index);
  }
  // push one last cellStartIndex for the last cell
  cellStartIndex.push_back(vector_index);

  // in case there are two boxes we need to remove the extra space allocated
  // here
  cellVector.resize(vector_index);
}

std::vector<std::vector<int>> CellList::GetNeighborList(uint box) const {
  return neighbors[box];
}

bool CellList::CompareCellList(CellList &other, int coordinateSize) {
  std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
  std::vector<int> otherCellVector, otherCellStartIndex, otherMapParticleToCell;

  for (uint box = 0; box < BOX_TOTAL; box++) {
    cellVector.resize(coordinateSize);
    cellStartIndex.resize(head[box].size());
    mapParticleToCell.resize(coordinateSize);

    otherCellVector.resize(coordinateSize);
    otherCellStartIndex.resize(head[box].size());
    otherMapParticleToCell.resize(coordinateSize);
  }

  for (uint box = 0; box < BOX_TOTAL; box++) {
    int vector_index = 0;
    for (size_t cell = 0; cell < head[box].size(); cell++) {
      cellStartIndex[cell] = vector_index;
      int particleIndex = head[box][cell];
      while (particleIndex != END_CELL) {
        cellVector[vector_index] = particleIndex;
        mapParticleToCell[particleIndex] = cell;
        vector_index++;
        particleIndex = list[particleIndex];
      }
    }
  }

  for (uint box = 0; box < BOX_TOTAL; box++) {
    int vector_index = 0;
    for (size_t cell = 0; cell < other.head[box].size(); cell++) {
      otherCellStartIndex[cell] = vector_index;
      int particleIndex = other.head[box][cell];
      while (particleIndex != END_CELL) {
        otherCellVector[vector_index] = particleIndex;
        otherMapParticleToCell[particleIndex] = cell;
        vector_index++;
        particleIndex = other.list[particleIndex];
      }
    }
  }

  if (list.size() == other.list.size()) {
    for (size_t i = 0; i < list.size(); i++) {
      if (list[i] != other.list[i])
        std::cout << "List objects are different" << std::endl;
    }
  }

  for (size_t i = 0; i < mapParticleToCell.size(); i++) {
    if (mapParticleToCell[i] != otherMapParticleToCell[i])
      return false;
  }

  std::cout << "CellList objects have equal states" << std::endl;

  return true;
}

void CellList::PrintList() {
  for (size_t i = 0; i < list.size(); i++)
    std::cout << list[i] << std::endl;

  std::cout << "head vector" << std::endl;
  for (int i = 0; i < BOX_TOTAL; i++) {
    for (size_t j = 0; j < head[i].size(); j++) {
      std::cout << head[i][j] << std::endl;
    }
  }
}
