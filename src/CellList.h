/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef CELLLIST_H
#define CELLLIST_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "BasicTypes.h"
#include "BoxDimensions.h"
#include "BoxDimensionsNonOrth.h"
#include "EnsemblePreprocessor.h"

class Molecules;
class XYZArray;
class BoxDimensions;
class MoleculeLookup;

//
// How finely the cell grid is cut, relative to the cutoff.
//
// 1 reproduces the historical behaviour exactly: cells about one cutoff wide
// and a +-1 stencil. Higher values cut finer cells and widen the stencil to
// match, which encloses less dead volume -- see ChooseStencilRadius.
//
// Pinned to 1 for CUDA builds: both GPU kernels index their flattened
// neighbour list with a hard-coded stride of 27
// (NUMBER_OF_NEIGHBOR_CELL in CalculateEnergyCUDAKernel.cu and
// CalculateForceCUDAKernel.cu), so a wider stencil would silently read the
// wrong cells.
//
#ifndef GOMC_CELL_REFINE
#define GOMC_CELL_REFINE 1
#endif
#ifdef GOMC_CUDA
#undef GOMC_CELL_REFINE
#define GOMC_CELL_REFINE 1
#endif

class CellList {
public:
  //
  // Cells along one box edge, for a given cutoff. The single definition of
  // the rule -- ResizeGrid applies it, and BoxDimensions::Init uses it to
  // report the resulting grid in the setup header.
  //
  // The clamp to three is load bearing. At three cells per edge the +-1
  // stencil RebuildNeighbors builds reaches all 27 cells, i.e. the whole box,
  // so the cell list prunes nothing while still paying its build and
  // traversal cost. CalculateEnergy detects that case (StencilCoversBox) and
  // runs a separate whole-box path instead. It holds whenever
  // side < 4 * cutoff, and the cutoff here is max(Rcut, RcutCoulomb) for the
  // box, not Rcut.
  //
  static int EdgeCellsFor(const double side, const double cutoff) {
    return EdgeCellsFor(side, cutoff, 1);
  }

  //
  // Cells along one edge for a stencil of radius r.
  //
  // Two constraints, both load bearing:
  //   correctness  r * cellSize >= cutoff, so the stencil reaches the whole
  //                cutoff sphere. cellSize = side / eCells, hence
  //                eCells <= r * side / cutoff.
  //   no aliasing  eCells >= 2r + 1, or the wrap-around in RebuildNeighbors
  //                lists the same cell twice and pairs get counted twice.
  //
  // Both hold only when side >= cutoff * (2 + 1/r), so a box can be too small
  // for a given radius; ChooseStencilRadius falls back in that case.
  //
  static int EdgeCellsFor(const double side, const double cutoff,
                          const int r) {
    return std::max((int)std::floor(side * r / cutoff), 2 * r + 1);
  }

  //
  // The largest usable stencil radius, up to maxR.
  //
  // A bigger radius means smaller cells, and smaller cells mean the stencil
  // encloses less dead volume: at 10K box 0 (70 A, cutoff 12) radius 1 gives
  // 14 A cells whose 27-cell stencil encloses 74,088 A^3 to find pairs inside
  // 7,238 A^3 -- measured, 10.03% of candidates survive. Radius 2 gives 6.36 A
  // cells and a 125-cell stencil of 32,186 A^3, about 22%.
  //
  // It is not free: finer cells hold fewer atoms, so the contiguous runs that
  // pass 1 vectorises over get shorter and the per-range overhead is paid more
  // often. Which way that trades is measured, not assumed.
  //
  static int ChooseStencilRadius(const double sx, const double sy,
                                 const double sz, const double cutoff,
                                 const int maxR) {
    // A box that is already degenerate at radius 1 -- grid clamped to 3 per
    // edge -- is left alone. CalculateEnergy bypasses the cell list entirely
    // for those (StencilCoversBox), and refining one just scatters a handful
    // of atoms over hundreds of mostly empty cells: 10K box 1 is 24 molecules
    // in a 300 A box, which radius 2 would spread over 216 cells and then
    // walk 125 ranges per atom to find nothing.
    if ((int)std::floor(sx / cutoff) <= 3 && (int)std::floor(sy / cutoff) <= 3 &&
        (int)std::floor(sz / cutoff) <= 3)
      return 1;
    for (int r = maxR; r > 1; --r) {
      const int need = 2 * r + 1;
      if ((int)std::floor(sx * r / cutoff) >= need &&
          (int)std::floor(sy * r / cutoff) >= need &&
          (int)std::floor(sz * r / cutoff) >= need)
        return r;
    }
    return 1;
  }

  //! Cells in this box's stencil. Equal to CellsInBox when the stencil covers
  //! the whole box, which is the case CalculateEnergy bypasses the cell list
  //! for entirely.
  int StencilCells(int box) const { return (int)neighbors[box][0].size(); }

  explicit CellList(const Molecules &mols, BoxDimensions &dims);
  CellList(const CellList &other);
  void SetCutoff();

  void RemoveMol(const int molIndex, const int box, const XYZArray &pos);
  void AddMol(const int molIndex, const int box, const XYZArray &pos);
  void GridAll(BoxDimensions &dims, const XYZArray &pos,
               const MoleculeLookup &lookup);
  void GridBox(BoxDimensions &dims, const XYZArray &pos,
               const MoleculeLookup &lookup, const uint b);
  void GetCellListNeighbor(uint box, int coordinateSize,
                           std::vector<int> &cellVector,
                           std::vector<int> &cellStartIndex,
                           std::vector<int> &mapParticleToCell) const;
  std::vector<std::vector<int>> GetNeighborList(uint box) const;

  // Index of cell containing position
  int PositionToCell(const XYZ &posRef, int box) const;

  // Iterates over all particles in a cell
  class Cell;
  Cell EnumerateCell(int cell, int box) const;

  // Iterates over all particles in the neighborhood of a cell
  class Neighbors;
  Neighbors EnumerateLocal(const XYZ &pos, int box) const;
  Neighbors EnumerateLocal(int cell, int box) const;

  // Iterates over all distinct, colocal pairs in a box
  class Pairs;
  Pairs EnumeratePairs(int box) const;

  int CellsInBox(int box) const { return head[box].size(); }

  //! Bumped by every mutator below. Lets a consumer cache something derived
  //! from the list contents and know when it went stale. Private setter by
  //! design: the increments live inside AddMol/RemoveMol/GridAll/GridBox so a
  //! caller cannot forget one. A missed bump gives silently wrong energies,
  //! not a crash, so nothing outside this class may touch it.
  unsigned Version() const { return version; }

  // true if every particle is a member of exactly one cell
  bool IsExhaustive() const;

  // GJS - Compare this cell list with another for evaluating parallel tempering
  // correctness
  bool CompareCellList(CellList &other, int coordinateSize);
  void PrintList();

  std::vector<int> list;
  std::vector<std::vector<int>> neighbors[BOX_TOTAL];
  std::vector<int> head[BOX_TOTAL];

private:
  static const int END_CELL = -1;
  unsigned version = 0;

  // Resize all boxes to match current axes
  void ResizeGrid(const BoxDimensions &dims);
  // Resize one boxes to match current axes
  void ResizeGridBox(const BoxDimensions &dims, const uint b);
  // Rebuild head/neighbor lists in box b to match current grid
  void RebuildNeighbors(int b);

  XYZ cellSize[BOX_TOTAL];
  int edgeCells[BOX_TOTAL][3];
  int stencilRadius[BOX_TOTAL] = {};
  const Molecules *mols;
  BoxDimensions *dimensions;
  double cutoff[BOX_TOTAL];
  bool isBuilt;
};

inline int CellList::PositionToCell(const XYZ &posRef, int box) const {
  // Transfer to unslant coordinate to find the neighbor
  XYZ pos = dimensions->TransformUnSlant(posRef, box);
  int x = (int)(pos.x / cellSize[box].x);
  int y = (int)(pos.y / cellSize[box].y);
  int z = (int)(pos.z / cellSize[box].z);
  // Check the cell number to avoid segfaults for coordinates close to axis
  // x, y, and z should never be equal or greater than number of cells in x, y,
  // and z axis, respectively.
  x -= (x == edgeCells[box][0] ? 1 : 0);
  y -= (y == edgeCells[box][1] ? 1 : 0);
  z -= (z == edgeCells[box][2] ? 1 : 0);
  return x * edgeCells[box][1] * edgeCells[box][2] + y * edgeCells[box][2] + z;
}

class CellList::Cell {
public:
  Cell(int start, const std::vector<int> &list)
      : at(start), list(list.begin()) {}

  int operator*() const { return at; }

  void Next() { at = list[at]; }

  bool Done() { return at == CellList::END_CELL; }

  void Jump(int head) { at = head; }

private:
  int at;
  std::vector<int>::const_iterator list;
};

class CellList::Neighbors {
public:
  Neighbors(const std::vector<int> &list, const std::vector<int> &head,
            const std::vector<int> &neighbors);

  int operator*() const { return *cell; }

  bool Done() const { return (neighbor == nEnd); }

  void Next();

private:
  CellList::Cell cell;
  std::vector<int>::const_iterator head;
  std::vector<int>::const_iterator neighbor, nEnd;
};

inline CellList::Cell CellList::EnumerateCell(int cell, int box) const {
#ifndef NDEBUG
  if (cell >= static_cast<int>(head[box].size())) {
    std::cout << "CellList.h:153: box " << box << ", Out of cell" << std::endl;
  }
#endif
  return CellList::Cell(head[box][cell], list);
}

inline CellList::Neighbors CellList::EnumerateLocal(int cell, int box) const {
#ifndef NDEBUG
  if (cell >= static_cast<int>(head[box].size())) {
    std::cout << "CellList.h:162: box " << box << ", Out of cell" << std::endl;
    std::cout << "AxisDimensions: " << dimensions->GetAxis(box) << std::endl;
  }
#endif
  return CellList::Neighbors(list, head[box], neighbors[box][cell]);
}

inline CellList::Neighbors CellList::EnumerateLocal(const XYZ &pos,
                                                    int box) const {
  int cell = PositionToCell(pos, box);
#ifndef NDEBUG
  if (cell >= static_cast<int>(head[box].size())) {
    std::cout << "CellList.h:172: box " << box << ", pos: " << pos << std::endl;
    std::cout << "AxisDimensions: " << dimensions->GetAxis(box) << std::endl;
  }
#endif
  return EnumerateLocal(cell, box);
}

inline CellList::Neighbors::Neighbors(const std::vector<int> &partList,
                                      const std::vector<int> &headList,
                                      const std::vector<int> &neighbors)
    : cell(headList[neighbors[0]], partList), head(headList.begin()),
      neighbor(neighbors.begin()), nEnd(neighbors.end()) {
  while (cell.Done()) {
    ++neighbor;
    if (Done()) {
      break;
    } else {
      cell.Jump(head[*neighbor]);
    }
  }
}

inline void CellList::Neighbors::Next() {
  cell.Next();
  // skip over empty cells
  while (cell.Done()) {
    ++neighbor;
    if (Done()) {
      break;
    } else {
      cell.Jump(head[*neighbor]);
    }
  }
  assert(!cell.Done() || Done());
}

class CellList::Pairs {
public:
  Pairs(const CellList &cellList, int box);

  int First() const { return *cellParticle; }
  int Second() const { return *localParticle; }
  void Next();
  bool Done() const { return cell == nCells; }

private:
  // skip to next nonempty cell
  void NextCell();

  CellList::Cell cellParticle;
  CellList::Neighbors localParticle;
  const CellList &cellList;
  int box, cell, nCells;
};

inline CellList::Pairs::Pairs(const CellList &cellList, int box)
    : cellParticle(cellList.EnumerateCell(0, box)),
      localParticle(cellList.EnumerateLocal(0, box)), cellList(cellList),
      box(box), cell(0), nCells(cellList.CellsInBox(box)) {
  if (cellParticle.Done())
    NextCell();
  if (First() >= Second() &&
      !(First() == CellList::END_CELL && Second() == CellList::END_CELL))
    Next();
}

inline void CellList::Pairs::NextCell() {
  do {
    ++cell;
    if (cell >= nCells)
      return;
    cellParticle = cellList.EnumerateCell(cell, box);
    localParticle = cellList.EnumerateLocal(cell, box);
    // skip empty cells
  } while (cellParticle.Done());
}

inline void CellList::Pairs::Next() {
  do {
    cellParticle.Next();
    if (cellParticle.Done()) {
      localParticle.Next();
      if (localParticle.Done()) {
        NextCell();
        if (Done())
          return;
      } else {
        cellParticle = cellList.EnumerateCell(cell, box);
      }
    }
    // skip over doubles
  } while (First() >= Second());
}
#endif /*CELLLIST_H*/
