/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef BOX_DIMENSIONS_NONORTHO_H
#define BOX_DIMENSIONS_NONORTHO_H

#include "BoxDimensions.h"

class BoxDimensionsNonOrth : public BoxDimensions {
public:
  BoxDimensionsNonOrth() : BoxDimensions() {
    cellLength.Init(BOX_TOTAL);
    for (uint b = 0; b < BOX_TOTAL; b++) {
      cellBasis[b] = XYZArray(3);
      cellBasis_Inv[b] = XYZArray(3);
    }
  }
  BoxDimensionsNonOrth(BoxDimensionsNonOrth const &other)
      : BoxDimensions(other) {
    cellLength.Init(BOX_TOTAL);
    other.cellLength.CopyRange(cellLength, 0, 0, BOX_TOTAL);
    for (uint b = 0; b < BOX_TOTAL; ++b) {
      cellBasis_Inv[b] = XYZArray(3);
      other.cellBasis_Inv[b].CopyRange(cellBasis_Inv[b], 0, 0, 3);
    }
  }

  ~BoxDimensionsNonOrth() {};

  BoxDimensionsNonOrth &operator=(BoxDimensionsNonOrth const &other);
  bool operator==(BoxDimensionsNonOrth const &other);
  // moved from Boxdimensions.h to support templates
  //  Returns if within cutoff, if it is, gets distance --
  //  with shortcut, same coordinate array
  bool InRcut(double &distSq, XYZ &dist, XYZArray const &arr, const uint i,
              const uint j, const uint b) const;

  // Dist squared -- with shortcut, two different coordinate arrays
  bool InRcut(double &distSq, XYZ &dist, XYZArray const &arr1, const uint i,
              XYZArray const &arr2, const uint j, const uint b) const;

  // Returns if within cutoff, if it is, gets distance --
  // with shortcut, same coordinate array
  bool InRcut(double &distSq, XYZArray const &arr, const uint i, const uint j,
              const uint b) const;

  // Dist squared -- with shortcut, two different coordinate arrays
  bool InRcut(double &distSq, XYZArray const &arr1, const uint i,
              XYZArray const &arr2, const uint j, const uint b) const;

  //! Non-orthogonal counterpart of BoxDimensions::DistSqRange. Same operation
  //! order as MinImage(): unslant, minimum image, slant back.
  void DistSqRange(double *__restrict distSq, const double xi, const double yi,
                   const double zi, const double *__restrict xj,
                   const double *__restrict yj, const double *__restrict zj,
                   const int m, const uint b) const {
    const XYZ i0 = cellBasis_Inv[b].Get(0), i1 = cellBasis_Inv[b].Get(1),
              i2 = cellBasis_Inv[b].Get(2);
    const XYZ c0 = cellBasis[b].Get(0), c1 = cellBasis[b].Get(1),
              c2 = cellBasis[b].Get(2);
    const double axX = axis.x[b], axY = axis.y[b], axZ = axis.z[b];
    const double hX = halfAx.x[b], hY = halfAx.y[b], hZ = halfAx.z[b];
    for (int k = 0; k < m; ++k) {
      const double rx = xi - xj[k], ry = yi - yj[k], rz = zi - zj[k];
      // TransformUnSlant
      double ux = rx * i0.x + ry * i1.x + rz * i2.x;
      double uy = rx * i0.y + ry * i1.y + rz * i2.y;
      double uz = rx * i0.z + ry * i1.z + rz * i2.z;
      // BoxDimensions::MinImage
      ux = MinImageSigned(ux, axX, hX);
      uy = MinImageSigned(uy, axY, hY);
      uz = MinImageSigned(uz, axZ, hZ);
      // TransformSlant
      const double sx = ux * c0.x + uy * c1.x + uz * c2.x;
      const double sy = ux * c0.y + uy * c1.y + uz * c2.y;
      const double sz = ux * c0.z + uy * c1.z + uz * c2.z;
      distSq[k] = sx * sx + sy * sy + sz * sz;
    }
  }

  void Init(config_setup::RestartSettings const &restart,
            config_setup::Volume const &confVolume,
            pdb_setup::Cryst1 const &cryst, Forcefield const &ff) override;

  void SetVolume(const uint b, const double vol) override;

  uint ShiftVolume(BoxDimensionsNonOrth &newDim, XYZ &scale, const uint b,
                   const double delta) const;

  //! Calculate and execute volume exchange based on transfer
  uint ExchangeVolume(BoxDimensionsNonOrth &newDim, XYZ *scale,
                      const double transfer, const uint *box) const;

  // Construct cell basis based on new axis dimension
  void CalcCellDimensions(const uint b);

  // Vector btwn two points, accounting for PBC, on an individual axis
  XYZ MinImage(XYZ rawVecRef, const uint b) const override;

  // Apply PBC, on X axis
  XYZ MinImage_X(XYZ rawVec, const uint b) const override;
  // Apply PBC, on Y axis
  XYZ MinImage_Y(XYZ rawVec, const uint b) const override;
  // Apply PBC, on Z axis
  XYZ MinImage_Z(XYZ rawVec, const uint b) const override;

  // Wrap one coordinate
  void WrapPBC(double &x, double &y, double &z, const uint b) const override;

  // Wrap one coordinate and check for PBC
  void WrapPBC(double &x, double &y, double &z, const uint b, const bool &pbcX,
               const bool &pbcY, const bool &pbcZ) const override;

  // Unwrap one coordinate
  void UnwrapPBC(double &x, double &y, double &z, const uint b,
                 XYZ const &ref) const override;

  // Transform A to unslant coordinate
  XYZ TransformUnSlant(const XYZ &A, const uint b) const override;

  // Transform A to slant coordinate
  XYZ TransformSlant(const XYZ &A, const uint b) const override;

  // private:
  XYZArray cellBasis_Inv[BOX_TOTAL]; // inverse cell matrix for each box
  XYZArray cellLength;               // Length of a, b, c for each box
};

// Calculate inverse transform
inline XYZ BoxDimensionsNonOrth::TransformUnSlant(const XYZ &A,
                                                  const uint b) const {
  XYZ temp;

  temp.x = A.x * cellBasis_Inv[b].Get(0).x + A.y * cellBasis_Inv[b].Get(1).x +
           A.z * cellBasis_Inv[b].Get(2).x;
  temp.y = A.x * cellBasis_Inv[b].Get(0).y + A.y * cellBasis_Inv[b].Get(1).y +
           A.z * cellBasis_Inv[b].Get(2).y;
  temp.z = A.x * cellBasis_Inv[b].Get(0).z + A.y * cellBasis_Inv[b].Get(1).z +
           A.z * cellBasis_Inv[b].Get(2).z;
  return temp;
}

// Calculate transform
inline XYZ BoxDimensionsNonOrth::TransformSlant(const XYZ &A,
                                                const uint b) const {
  XYZ temp;

  temp.x = A.x * cellBasis[b].Get(0).x + A.y * cellBasis[b].Get(1).x +
           A.z * cellBasis[b].Get(2).x;
  temp.y = A.x * cellBasis[b].Get(0).y + A.y * cellBasis[b].Get(1).y +
           A.z * cellBasis[b].Get(2).y;
  temp.z = A.x * cellBasis[b].Get(0).z + A.y * cellBasis[b].Get(1).z +
           A.z * cellBasis[b].Get(2).z;
  return temp;
}

inline bool BoxDimensionsNonOrth::InRcut(double &distSq, XYZ &dist,
                                         XYZArray const &arr, const uint i,
                                         const uint j, const uint b) const {
  dist = BoxDimensionsNonOrth::MinImage(arr.Difference(i, j), b);
  distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
  return (rCutSq[b] > distSq);
}

inline bool BoxDimensionsNonOrth::InRcut(double &distSq, XYZ &dist,
                                         XYZArray const &arr1, const uint i,
                                         XYZArray const &arr2, const uint j,
                                         const uint b) const {
  dist = BoxDimensionsNonOrth::MinImage(arr1.Difference(i, arr2, j), b);
  distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
  return (rCutSq[b] > distSq);
}

inline bool BoxDimensionsNonOrth::InRcut(double &distSq, XYZArray const &arr,
                                         const uint i, const uint j,
                                         const uint b) const {
  XYZ dist = BoxDimensionsNonOrth::MinImage(arr.Difference(i, j), b);
  distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
  return (rCutSq[b] > distSq);
}

inline bool BoxDimensionsNonOrth::InRcut(double &distSq, XYZArray const &arr1,
                                         const uint i, XYZArray const &arr2,
                                         const uint j, const uint b) const {
  XYZ dist = BoxDimensionsNonOrth::MinImage(arr1.Difference(i, arr2, j), b);
  distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
  return (rCutSq[b] > distSq);
}

inline XYZ BoxDimensionsNonOrth::MinImage(XYZ rawVecRef, const uint b) const {
  XYZ rawVec = TransformUnSlant(rawVecRef, b);
  rawVecRef = BoxDimensions::MinImage(rawVec, b);
  rawVecRef = TransformSlant(rawVecRef, b);
  return rawVecRef;
}

inline XYZ BoxDimensionsNonOrth::MinImage_X(XYZ rawVecRef, const uint b) const {
  XYZ rawVec = TransformUnSlant(rawVecRef, b);
  rawVecRef = BoxDimensions::MinImage_X(rawVec, b);
  rawVecRef = TransformSlant(rawVecRef, b);
  return rawVecRef;
}

inline XYZ BoxDimensionsNonOrth::MinImage_Y(XYZ rawVecRef, const uint b) const {
  XYZ rawVec = TransformUnSlant(rawVecRef, b);
  rawVecRef = BoxDimensions::MinImage_Y(rawVec, b);
  rawVecRef = TransformSlant(rawVecRef, b);
  return rawVecRef;
}

inline XYZ BoxDimensionsNonOrth::MinImage_Z(XYZ rawVecRef, const uint b) const {
  XYZ rawVec = TransformUnSlant(rawVecRef, b);
  rawVecRef = BoxDimensions::MinImage_Z(rawVec, b);
  rawVecRef = TransformSlant(rawVecRef, b);
  return rawVecRef;
}

#endif /*BOX_DIMENSIONS_NONORTHO_H*/
