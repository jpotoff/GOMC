/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
Copyright (C) 2022 GOMC Group
A copy of the MIT License can be found in License.txt
along with this program, also can be found at
<https://opensource.org/licenses/MIT>.
********************************************************************************/
#ifndef FF_TABULATED_H
#define FF_TABULATED_H

#include "BasicTypes.h" //for uint
#include "FFConst.h"    //constants related to particles.
#include "FFParticle.h"
#include "NumLib.h" //For Cb, Sq
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef M_2_SQRTPI
#define M_2_SQRTPI (2.0 / sqrt(M_PI))
#endif

//////////////////////////////////////////////////////////////////////
////////////////////////// Tabulated Potential Style
///////////////////////////////
//////////////////////////////////////////////////////////////////////
// Tabulated potential calculation using NAMD-style file format:
// File format: distance energy force (one line per distance point)
// Energy and virial are computed via linear interpolation from the table
//
// Units:
//   - Distance: Angstroms (Å)
//   - Energy: K (Kelvin) for non-CHARMM, kcal/mol for CHARMM (converted to K)
//   - Force: K/Å for non-CHARMM, kcal/mol/Å for CHARMM (converted to K/Å)
//
// E(r) = interpolated from table
// Vir(r) = F(r) * r, where F(r) is interpolated from table
// F(r) = -dE/dr, so Vir(r) = -r * dE/dr  CalcVir returns Wij = F/r = -dE/dr
//
// When ParaTypeCHARMM=True, energy and force values are converted from
// kcal/mol and kcal/mol/Å to K and K/Å using KCAL_PER_MOL_TO_K constant.

struct TabulatedData {
  std::vector<double> distance; // Distance values
  std::vector<double> energy;   // Energy values
  std::vector<double> force;    // Force values (F = -dE/dr)
  double rMin;                  // Minimum distance in table
  double rMax;                  // Maximum distance in table
  double dr;                    // Spacing between points (if uniform)
  bool uniformSpacing;          // Whether spacing is uniform
  std::string pairType;         // Pair type identifier (e.g., "OO", "CH", etc.)

  TabulatedData()
      : rMin(0.0), rMax(0.0), dr(0.0), uniformSpacing(false), pairType("") {}
};

struct FF_TABULATED : public FFParticle {
public:
  FF_TABULATED(Forcefield &ff)
      : FFParticle(ff), tableData(NULL), tableData_1_4(NULL),
        nbtableData(NULL) {}

  virtual ~FF_TABULATED() {
    delete[] tableData;
    delete[] tableData_1_4;
  }

  virtual void Init(ff_setup::Particle const &mie,
                    ff_setup::NBfix const &nbfix);

  // Initialize with NBtable data available for lookup
  void InitWithNBtable(ff_setup::Particle const &mie,
                       ff_setup::NBfix const &nbfix,
                       const ff_setup::NBtable &nbtable) {
    // Set the NBtable data first, before calling Init
    nbtableData = &nbtable;
    // Now call the regular Init
    Init(mie, nbfix);
  }

  // Build mapping from (kind1, kind2) pairs to pair type names from NBtable
  // Uses the provided mie particle name list to resolve atom type strings to
  // kind indices. This avoids assuming NBtable entries are in the same
  // sequential order as the internal FlatIndex ordering.
  void BuildPairTypeMap(const ff_setup::Particle &mie) {
    pairTypeMap.clear();

    if (nbtableData == NULL) {
      throw std::runtime_error(
          "Fatal Error: FF_TABULATED initialized without NBtable data. "
          "Ensure InitWithNBtable is called instead of Init.");
    }

    std::cout << "Building Pair Type Map for Tabulated Potentials:"
              << std::endl;
    std::cout << "Total particle kinds: " << count << std::endl;
    std::cout << "Total NBtable entries: " << nbtableData->GetPairCount()
              << std::endl;

    // Print mie particle type names for debugging
    // std::cout << "Particle names (count=" << mie.getnamecnt() << "): ";
    for (uint k = 0; k < mie.getnamecnt(); ++k) {
      std::cout << "'" << mie.getname(k) << "'";
      if (k + 1 < mie.getnamecnt())
        std::cout << ", ";
    }
    std::cout << std::endl;

    if (nbtableData->GetPairCount() == 0) {
      throw std::runtime_error("Fatal Error: NBTable has no entries. Check "
                               "the tabulated potential file.");
    }

    // For each NBtable entry, find the matching kind indices from mie.getname()
    for (size_t t = 0; t < nbtableData->GetPairCount(); ++t) {
      std::string atom1 = nbtableData->GetAtomType1(t);
      std::string atom2 = nbtableData->GetAtomType2(t);
      std::string pairType = nbtableData->GetTablePairName(t);

      std::cout << "NBtable entry " << t << ": atoms=('" << atom1 << "','"
                << atom2 << "') -> pairType='" << pairType << "'" << std::endl;

      // Find kind indices for atom1 and atom2 in mie.getname list
      int kindA = -1, kindB = -1;
      for (uint k = 0; k < mie.getnamecnt(); ++k) {
        if (mie.getname(k) == atom1)
          kindA = static_cast<int>(k);
        if (mie.getname(k) == atom2)
          kindB = static_cast<int>(k);
        if (kindA != -1 && kindB != -1)
          break;
      }

      if (kindA == -1 || kindB == -1) {
        std::cout << "  (Available mie names: ";
        for (uint k = 0; k < mie.getnamecnt(); ++k) {
          std::cout << "'" << mie.getname(k) << "'";
          if (k + 1 < mie.getnamecnt())
            std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        throw std::runtime_error(
            "Fatal Error: Could not find atom types '" + atom1 + "' and '" +
            atom2 + "' in particle type list (pairType: '" + pairType + "')");
      }

      uint minKind =
          std::min(static_cast<uint>(kindA), static_cast<uint>(kindB));
      uint maxKind =
          std::max(static_cast<uint>(kindA), static_cast<uint>(kindB));

      std::stringstream key;
      key << minKind << "_" << maxKind;
      std::string keyStr = key.str();

      pairTypeMap[keyStr] = pairType;

      //  std::cout << "Mapping NBtable entry: atoms ('" << atom1 << "','"
      //            << atom2 << "') -> kinds (" << minKind << "," << maxKind
      //            << ") -> '" << pairType << "'" << std::endl;
    }

    std::cout << "Map Building Complete: " << pairTypeMap.size()
              << " entries in map" << std::endl;
    std::cout << std::endl;
    // Dump map contents for extra visibility
    // if (!pairTypeMap.empty()) {
    //  std::cout << "PairTypeMap contents:" << std::endl;
    //  for (auto &entry : pairTypeMap) {
    //    std::cout << "  '" << entry.first << "' -> '" << entry.second << "'"
    //              << std::endl;
    //  }
    //}
    // std::cout << std::endl;
  }

  // Method to register tabulated potential file for a specific pair type
  void RegisterPairTableFile(const uint kind1, const uint kind2,
                             const std::string &filename) {
    uint minKind = std::min(kind1, kind2);
    uint maxKind = std::max(kind1, kind2);

    std::stringstream ss;
    ss << minKind << "_" << maxKind;
    pairTableFiles[ss.str()] = filename;
  }

  // Method to register tabulated potential files from NBtable setup data
  void RegisterPairTableFiles(const ff_setup::NBtable &nbtable) {
    // Extract pair information from nbtable
    // The nbtable contains a list of pair names in format "type1_type2"
    for (size_t i = 0; i < nbtable.getnamecnt(); i++) {
      std::string pairName = nbtable.getname(i);
      // Parse "type1_type2" format
      size_t underscore = pairName.find('_');
      if (underscore != std::string::npos) {
        std::string kind1Str = pairName.substr(0, underscore);
        std::string kind2Str = pairName.substr(underscore + 1);

        try {
          uint kind1 = std::stoul(kind1Str);
          uint kind2 = std::stoul(kind2Str);

          std::cout << "Registered pair type (" << kind1 << ", " << kind2 << ")"
                    << std::endl;
        } catch (const std::exception &e) {
          std::cout << "Warning: Could not parse pair type from NBtable entry: "
                    << pairName << std::endl;
        }
      }
    }
  }

  virtual double CalcEn(const double distSq, const uint kind1, const uint kind2,
                        const double lambda) const;
  virtual double CalcVir(const double distSq, const uint kind1,
                         const uint kind2, const double lambda) const;
  virtual void CalcAdd_1_4(double &en, const double distSq, const uint kind1,
                           const uint kind2) const;

  // coulomb interaction functions
  virtual double CalcCoulomb(const double distSq, const uint kind1,
                             const uint kind2, const double qi_qj_Fact,
                             const double lambda, const uint b) const;
  virtual double CalcCoulombVir(const double distSq, const uint kind1,
                                const uint kind2, const double qi_qj,
                                const double lambda, const uint b) const;
  virtual void CalcCoulombAdd_1_4(double &en, const double distSq,
                                  const double qi_qj_Fact, const bool NB) const;

  //! Returns Ezero, no energy correction
  virtual double EnergyLRC(const uint kind1, const uint kind2) const {
    return 0.0;
  }
  //!!Returns Ezero, no virial correction
  virtual double VirialLRC(const uint kind1, const uint kind2) const {
    return 0.0;
  }
  //! Returns zero for impulse pressure correction term for a kind pair
  virtual double ImpulsePressureCorrection(const uint kind1,
                                           const uint kind2) const {
    return 0.0;
  }

  // Calculate the dE/dlambda for vdw energy
  virtual double CalcdEndL(const double distSq, const uint kind1,
                           const uint kind2, const double lambda) const;
  // Calculate the dE/dlambda for Coulomb energy
  virtual double CalcCoulombdEndL(const double distSq, const uint kind1,
                                  const uint kind2, const double qi_qj_Fact,
                                  const double lambda, uint b) const;

protected:
  virtual double CalcEn(const double distSq, const uint index) const;
  virtual double CalcVir(const double distSq, const uint index) const;
  virtual double CalcCoulomb(const double distSq, const double qi_qj_Fact,
                             const uint b) const;
  virtual double CalcCoulombVir(const double distSq, const double qi_qj,
                                uint b) const;

  // Helper functions for tabulated potential
  void ReadTabulatedFile(const std::string &filename,
                         TabulatedData &potentialTable, const bool isCHARMM);
  double InterpolateEnergy(const TabulatedData &data, const double dist) const;
  double InterpolateForce(const TabulatedData &data, const double dist) const;
  uint FindTableIndex(const TabulatedData &data, const double dist) const;
  std::string GetPairTypeString(const uint kind1, const uint kind2) const;

  // Storage for tabulated data
  TabulatedData *tableData;     // Tabulated data for each pair type
  TabulatedData *tableData_1_4; // Tabulated data for 1-4 interactions

  // Map to store per-pair tabulated potential file names
  // Key format: "type1_type2" where type1 <= type2
  std::map<std::string, std::string> pairTableFiles;

  // Reference to NBtable data for pair type lookups
  const ff_setup::NBtable *nbtableData;

  // Map from "(kind1_kind2)" to pair type name from NBtable
  std::map<std::string, std::string> pairTypeMap;
};

/**
 * @brief Read tabulated potential data from a file
 * @param filename Name of the file to read
 * @param potentialTable TabulatedData object to store the data
 * @param isCHARMM Whether the data is in CHARMM format
 * @note This function reads all tabulated potential data from a file (even if
 * not needed).
 * @note The data is stored "potentialTable.distance", "potentialTable.energy",
 * and "potentialTable.force".
 */
inline void FF_TABULATED::ReadTabulatedFile(const std::string &filename,
                                            TabulatedData &potentialTable,
                                            const bool isCHARMM) {
  std::ifstream file(filename.c_str());
  if (!file.is_open()) {
    std::cout << "ERROR: Cannot open tabulated potential file: " << filename
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  double r, e, f;
  bool foundPair = false;
  int dbgPrinted = 0;

  // Clear previous data
  potentialTable.distance.clear();
  potentialTable.energy.clear();
  potentialTable.force.clear();

  // Unit conversion factor for CHARMM format
  // Energy: kcal/mol -> K, Force: kcal/mol/Å -> K/Å
  double energyConversion = 1.0;
  double forceConversion = 1.0;
  if (isCHARMM) {
    energyConversion = ff_setup::KCAL_PER_MOL_TO_K;
    forceConversion = ff_setup::KCAL_PER_MOL_TO_K;
    std::cout
        << "DEBUG: Using CHARMM unit conversions for tabulated potential file: "
        << filename << std::endl;
  }

  // Read file line by line looking for the specified pair type
  while (std::getline(file, line)) {
    // Trim leading whitespace
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
      continue; // Empty line

    line = line.substr(start);

    // Check for TYPE keyword
    if (line.substr(0, 4) == "TYPE") {
      // If we were already reading data and now found a new TYPE, stop
      if (foundPair && !potentialTable.distance.empty()) {
        break;
      }

      // Parse the pair type from the line
      std::istringstream typeStream(line);
      std::string typeKeyword, typeStr;
      typeStream >> typeKeyword >> typeStr;

      // std::cout << "DEBUG: Found TYPE keyword with pair: '" << typeStr << "'"
      //           << " (looking for '" << potentialTable.pairType << "')" <<
      //           std::endl;

      // Check if this is the pair type we're looking for
      if (typeStr == potentialTable.pairType) {
        foundPair = true;
        //  std::cout << "Found pair type: " << typeStr << std::endl;
        continue;
      } else if (foundPair) {
        // We were reading this pair and now found a different one, so stop
        break;
      }
      continue;
    }

    // Skip comments (lines starting with # or !)
    if (line[0] == '#' || line[0] == '!')
      continue;

    // If we haven't found our pair type yet, skip this line
    if (!foundPair)
      continue;

    // Try to read three numbers: distance, energy, force
    std::istringstream iss(line);
    if (iss >> r >> e >> f) {
      potentialTable.distance.push_back(r);
      // Convert energy from kcal/mol to K if CHARMM format
      potentialTable.energy.push_back(e * energyConversion);
      // Convert force from kcal/mol/Å to K/Å if CHARMM format
      potentialTable.force.push_back(f * forceConversion);
    }
  }

  file.close();

  if (!foundPair) {
    std::cout << "ERROR: Pair type '" << potentialTable.pairType
              << "' not found in "
              << "tabulated potential file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  if (potentialTable.distance.empty()) {
    std::cout << "ERROR: No valid data found for pair type '"
              << potentialTable.pairType
              << "' in tabulated potential file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // Find min and max distances
  potentialTable.rMin = *std::min_element(potentialTable.distance.begin(),
                                          potentialTable.distance.end());
  potentialTable.rMax = *std::max_element(potentialTable.distance.begin(),
                                          potentialTable.distance.end());

  // Check if spacing is uniform
  if (potentialTable.distance.size() > 1) {
    potentialTable.dr = potentialTable.distance[1] - potentialTable.distance[0];
    potentialTable.uniformSpacing = true;
    for (size_t i = 2; i < potentialTable.distance.size(); i++) {
      double expected = potentialTable.distance[0] + i * potentialTable.dr;
      if (std::fabs(potentialTable.distance[i] - expected) > 1.0e-6) {
        potentialTable.uniformSpacing = false;
        break;
      }
    }
  }

  std::string unitStr = isCHARMM ? "kcal/mol (converted to K)" : "K";
  std::cout << "Info: Loaded tabulated potential for pair type '"
            << potentialTable.pairType << "' from " << filename << " with "
            << potentialTable.distance.size() << " points, range ["
            << potentialTable.rMin << ", " << potentialTable.rMax
            << "] Angstrom, energy units: " << unitStr << std::endl;
}

inline uint FF_TABULATED::FindTableIndex(const TabulatedData &data,
                                         const double dist) const {
  if (dist <= data.rMin)
    return 0;
  if (dist >= data.rMax)
    return data.distance.size() - 2; // Return second-to-last for interpolation

  // Binary search for non-uniform spacing, or direct calculation for uniform
  if (data.uniformSpacing) {
    uint idx = static_cast<uint>((dist - data.rMin) / data.dr);
    return std::min(idx, static_cast<uint>(data.distance.size() - 2));
  } else {
    // Binary search
    uint left = 0;
    uint right = data.distance.size() - 1;
    while (right - left > 1) {
      uint mid = (left + right) / 2;
      if (data.distance[mid] <= dist)
        left = mid;
      else
        right = mid;
    }
    return left;
  }
}

inline double FF_TABULATED::InterpolateEnergy(const TabulatedData &data,
                                              const double dist) const {
  // Safety check: if table is empty, return zero energy
  if (data.distance.empty()) {
    throw std::runtime_error(
        "Fatal Error: InterpolateEnergy called with empty "
        "table for pair type '" +
        data.pairType + "'. Check force field file for missing NBTable entry.");
  }

  if (dist <= data.rMin)
    return data.energy[0];
  if (dist >= data.rMax)
    return data.energy[data.energy.size() - 1];

  uint idx = FindTableIndex(data, dist);
  double r1 = data.distance[idx];
  double r2 = data.distance[idx + 1];
  double e1 = data.energy[idx];
  double e2 = data.energy[idx + 1];

  // Check interpolation type
  if (forcefield.interpolationType == "cubic" && idx >= 1 &&
      idx + 2 < data.distance.size()) {
    // Cubic interpolation using 4 points: idx-1, idx, idx+1, idx+2
    double x0 = data.distance[idx - 1];
    double x1 = r1;
    double x2 = r2;
    double x3 = data.distance[idx + 2];
    double y0 = data.energy[idx - 1];
    double y1 = e1;
    double y2 = e2;
    double y3 = data.energy[idx + 2];

    // Lagrange interpolation
    double term0 = y0 * (dist - x1) * (dist - x2) * (dist - x3) /
                   ((x0 - x1) * (x0 - x2) * (x0 - x3));
    double term1 = y1 * (dist - x0) * (dist - x2) * (dist - x3) /
                   ((x1 - x0) * (x1 - x2) * (x1 - x3));
    double term2 = y2 * (dist - x0) * (dist - x1) * (dist - x3) /
                   ((x2 - x0) * (x2 - x1) * (x2 - x3));
    double term3 = y3 * (dist - x0) * (dist - x1) * (dist - x2) /
                   ((x3 - x0) * (x3 - x1) * (x3 - x2));

    return term0 + term1 + term2 + term3;
  } else {
    // Linear interpolation
    double t = (dist - r1) / (r2 - r1);
    return e1 + t * (e2 - e1);
  }
}

inline double FF_TABULATED::InterpolateForce(const TabulatedData &data,
                                             const double dist) const {
  // Safety check: if table is empty, return zero force
  if (data.distance.empty()) {
    std::cout
        << "WARNING: InterpolateForce called with empty table for pair type '"
        << data.pairType << "'" << std::endl;
    return 0.0;
  }

  if (dist <= data.rMin)
    return data.force[0];
  if (dist >= data.rMax)
    return data.force[data.force.size() - 1];

  uint idx = FindTableIndex(data, dist);
  double r1 = data.distance[idx];
  double r2 = data.distance[idx + 1];
  double f1 = data.force[idx];
  double f2 = data.force[idx + 1];

  // Check interpolation type
  if (forcefield.interpolationType == "cubic" && idx >= 1 &&
      idx + 2 < data.distance.size()) {
    // Cubic interpolation using 4 points: idx-1, idx, idx+1, idx+2
    double x0 = data.distance[idx - 1];
    double x1 = r1;
    double x2 = r2;
    double x3 = data.distance[idx + 2];
    double y0 = data.force[idx - 1];
    double y1 = f1;
    double y2 = f2;
    double y3 = data.force[idx + 2];

    // Lagrange interpolation
    double term0 = y0 * (dist - x1) * (dist - x2) * (dist - x3) /
                   ((x0 - x1) * (x0 - x2) * (x0 - x3));
    double term1 = y1 * (dist - x0) * (dist - x2) * (dist - x3) /
                   ((x1 - x0) * (x1 - x2) * (x1 - x3));
    double term2 = y2 * (dist - x0) * (dist - x1) * (dist - x3) /
                   ((x2 - x0) * (x2 - x1) * (x2 - x3));
    double term3 = y3 * (dist - x0) * (dist - x1) * (dist - x2) /
                   ((x3 - x0) * (x3 - x1) * (x3 - x2));

    return term0 + term1 + term2 + term3;
  } else {
    // Linear interpolation
    double t = (dist - r1) / (r2 - r1);
    return f1 + t * (f2 - f1);
  }
}

inline void FF_TABULATED::Init(ff_setup::Particle const &mie,
                               ff_setup::NBfix const &nbfix) {
  // Initialize sigma and epsilon (needed for compatibility)
  FFParticle::Init(mie, nbfix);

  // Build the pair type map NOW that count is set and nbtableData is available
  if (nbtableData != NULL) {
    BuildPairTypeMap(mie);
  } else {
    std::cout << "WARNING: nbtableData is NULL in Init()" << std::endl;
  }

  uint size = num::Sq(count);
  tableData = new TabulatedData[size];
  tableData_1_4 = new TabulatedData[size];

  // Check if tabulated potential file is specified
  if (forcefield.tabulatedPotentialFile.empty()) {
    std::cout << "ERROR: Tabulated potential file not specified!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize all tables with data from the appropriate files
  // For each pair type (i, j), get the corresponding table from the
  // consolidated file Note: We iterate ALL pairs (i,j) and use the symmetric
  // mapping since FlatIndex(i,j) can be called with any order during runtime.
  for (uint i = 0; i < count; i++) {
    for (uint j = 0; j < count; j++) {
      uint index = FlatIndex(i, j);

      // Check if this pair exists in the pairTypeMap using min/max ordering
      uint minKind = std::min(i, j);
      uint maxKind = std::max(i, j);
      std::stringstream keyStream;
      keyStream << minKind << "_" << maxKind;
      std::string key = keyStream.str();

      std::map<std::string, std::string>::const_iterator it =
          pairTypeMap.find(key);
      if (it == pairTypeMap.end()) {
        std::cout << "Warning: Pair (" << i << ", " << j << ") [key '" << key
                  << "'] not found in pairTypeMap; skipping table load"
                  << std::endl;
        // Initialize with descriptive name to aid debugging if accessed
        tableData[index].pairType =
            "UnknownPair: " + mie.getname(i) + "_" + mie.getname(j);
        tableData[index].rMin = 0.0;
        tableData[index].rMax = 0.0;
        tableData[index].uniformSpacing = false;
        tableData_1_4[index] = tableData[index];
        continue;
      }

      // Construct the pair type string (e.g., "OO", "CH", etc.)
      // This should match the TYPE keyword in the tabulated file
      std::string pairType = it->second;
      tableData[index].pairType = pairType;

      // Read the tabulated potential data for this pair from the consolidated
      // file Only read once (check if already loaded to avoid re-reading)
      if (tableData[index].distance.empty()) {
        ReadTabulatedFile(forcefield.tabulatedPotentialFile, tableData[index],
                          forcefield.isCHARMM);
      }

      // For 1-4 interactions, use the same table as regular interactions
      tableData_1_4[index] = tableData[index];
    }
  }
}

// Helper function to get the pair type string from atom type indices
inline std::string FF_TABULATED::GetPairTypeString(const uint kind1,
                                                   const uint kind2) const {
  // Normalize the pair (use min, max order)
  uint minKind = std::min(kind1, kind2);
  uint maxKind = std::max(kind1, kind2);

  std::stringstream keyStream;
  keyStream << minKind << "_" << maxKind;
  std::string key = keyStream.str();

  std::cout << "DEBUG GetPairTypeString: Looking for key '" << key
            << "' in map with " << pairTypeMap.size() << " entries"
            << std::endl;

  // Look up in the pre-built map
  std::map<std::string, std::string>::const_iterator it = pairTypeMap.find(key);
  if (it != pairTypeMap.end()) {
    std::cout << "  FOUND: Pair (" << kind1 << ", " << kind2 << ") -> '"
              << it->second << "'" << std::endl;
    return it->second;
  }

  // Fallback: return indices as string if map lookup fails
  std::cout << "  NOT FOUND: Pair (" << kind1 << ", " << kind2
            << ") not in map, using fallback" << std::endl;

  // Print map contents for debugging
  if (pairTypeMap.empty()) {
    std::cout << "  Map is EMPTY!" << std::endl;
  } else {
    std::cout << "  Map contains:" << std::endl;
    for (auto &entry : pairTypeMap) {
      std::cout << "    '" << entry.first << "' -> '" << entry.second << "'"
                << std::endl;
    }
  }

  return key;
}

inline double FF_TABULATED::CalcEn(const double distSq, const uint kind1,
                                   const uint kind2,
                                   const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  if (lambda >= 0.999999) {
    return CalcEn(distSq, index);
  }

  // For lambda scaling, use soft-core approach similar to other force fields
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);
  double dist = sqrt(softRsq);

  double en = InterpolateEnergy(tableData[index], dist);
  return lambda * en;
}

inline double FF_TABULATED::CalcEn(const double distSq,
                                   const uint index) const {
  double dist = sqrt(distSq);
  return InterpolateEnergy(tableData[index], dist);
}

inline double FF_TABULATED::CalcVir(const double distSq, const uint kind1,
                                    const uint kind2,
                                    const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  if (lambda >= 0.999999) {
    return CalcVir(distSq, index);
  }

  // For lambda scaling
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);
  double dist = sqrt(softRsq);
  double correction = distSq / softRsq;

  double force = InterpolateForce(tableData[index], dist);
  // Use the same soft-core scaling as other forcefields: scale the per-index
  // virial (which is F/r) by correction^2 and lambda
  double vir = lambda * correction * correction * CalcVir(softRsq, index);
  return vir;
}

inline double FF_TABULATED::CalcVir(const double distSq,
                                    const uint index) const {
  double dist = sqrt(distSq);
  double force = InterpolateForce(tableData[index], dist);
  // CalcVir should return the scalar Wij = (-dE/dr)/r so that
  // r_vector * Wij = force vector. Since InterpolateForce returns
  // F = -dE/dr, we return Wij = F / r.
  if (dist > 0.0)
    return force / dist;
  else
    return 0.0;
}

inline void FF_TABULATED::CalcAdd_1_4(double &en, const double distSq,
                                      const uint kind1,
                                      const uint kind2) const {
  if (forcefield.rCutSq < distSq)
    return;

  uint index = FlatIndex(kind1, kind2);
  double dist = sqrt(distSq);
  en += InterpolateEnergy(tableData_1_4[index], dist);
}

inline double FF_TABULATED::CalcCoulomb(const double distSq, const uint kind1,
                                        const uint kind2,
                                        const double qi_qj_Fact,
                                        const double lambda,
                                        const uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  if (lambda >= 0.999999) {
    return CalcCoulomb(distSq, qi_qj_Fact, b);
  }

  double en = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    en = lambda * CalcCoulomb(softRsq, qi_qj_Fact, b);
  } else {
    en = lambda * CalcCoulomb(distSq, qi_qj_Fact, b);
  }
  return en;
}

inline double FF_TABULATED::CalcCoulomb(const double distSq,
                                        const double qi_qj_Fact,
                                        const uint b) const {
  if (forcefield.ewald) {
    double dist = sqrt(distSq);
    double val = forcefield.alpha[b] * dist;
    return qi_qj_Fact * erfc(val) / dist;
  } else {
    double dist = sqrt(distSq);
    return qi_qj_Fact / dist;
  }
}

inline double FF_TABULATED::CalcCoulombVir(const double distSq,
                                           const uint kind1, const uint kind2,
                                           const double qi_qj,
                                           const double lambda,
                                           const uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  if (lambda >= 0.999999) {
    return CalcCoulombVir(distSq, qi_qj, b);
  }

  double vir = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    double correction = distSq / softRsq;
    vir = lambda * correction * correction * CalcCoulombVir(softRsq, qi_qj, b);
  } else {
    vir = lambda * CalcCoulombVir(distSq, qi_qj, b);
  }
  return vir;
}

inline double FF_TABULATED::CalcCoulombVir(const double distSq,
                                           const double qi_qj,
                                           const uint b) const {
  if (forcefield.ewald) {
    double dist = sqrt(distSq);
    double constValue = forcefield.alpha[b] * M_2_SQRTPI;
    double expConstValue = exp(-1.0 * forcefield.alphaSq[b] * distSq);
    double temp = erfc(forcefield.alpha[b] * dist);
    return qi_qj * (temp / dist + constValue * expConstValue) / distSq;
  } else {
    double dist = sqrt(distSq);
    return qi_qj / (distSq * dist);
  }
}

inline void FF_TABULATED::CalcCoulombAdd_1_4(double &en, const double distSq,
                                             const double qi_qj_Fact,
                                             const bool NB) const {
  if (forcefield.rCutSq < distSq)
    return;

  double dist = sqrt(distSq);
  if (NB)
    en += qi_qj_Fact / dist;
  else
    en += qi_qj_Fact * forcefield.scaling_14 / dist;
}

inline double FF_TABULATED::CalcdEndL(const double distSq, const uint kind1,
                                      const uint kind2,
                                      const double lambda) const {
  if (forcefield.rCutSq < distSq)
    return 0.0;

  uint index = FlatIndex(kind1, kind2);
  double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
  sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
  double dist6 = distSq * distSq * distSq;
  double lambdaCoef =
      forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
  double softDist6 = lambdaCoef * sigma6 + dist6;
  double softRsq = cbrt(softDist6);
  double dist = sqrt(softRsq);

  double fCoef = lambda * forcefield.sc_alpha * forcefield.sc_power / 6.0;
  fCoef *= pow(1.0 - lambda, forcefield.sc_power - 1.0) * sigma6 /
           (softRsq * softRsq);

  double en = InterpolateEnergy(tableData[index], dist);
  double force = InterpolateForce(tableData[index], dist);
  // CalcVir(index) returns Wij = F/r, so use vir = Wij
  double vir = 0.0;
  if (dist > 0.0)
    vir = force / dist;

  double dhdl = en + fCoef * vir;
  return dhdl;
}

inline double FF_TABULATED::CalcCoulombdEndL(const double distSq,
                                             const uint kind1, const uint kind2,
                                             const double qi_qj_Fact,
                                             const double lambda,
                                             uint b) const {
  if (forcefield.rCutCoulombSq[b] < distSq)
    return 0.0;

  double dhdl = 0.0;
  if (forcefield.sc_coul) {
    uint index = FlatIndex(kind1, kind2);
    double sigma6 = sigmaSq[index] * sigmaSq[index] * sigmaSq[index];
    sigma6 = std::max(sigma6, forcefield.sc_sigma_6);
    double dist6 = distSq * distSq * distSq;
    double lambdaCoef =
        forcefield.sc_alpha * pow((1.0 - lambda), forcefield.sc_power);
    double softDist6 = lambdaCoef * sigma6 + dist6;
    double softRsq = cbrt(softDist6);
    double fCoef = lambda * forcefield.sc_alpha * forcefield.sc_power / 6.0;
    fCoef *= pow(1.0 - lambda, forcefield.sc_power - 1.0) * sigma6 /
             (softRsq * softRsq);
    dhdl = CalcCoulomb(softRsq, qi_qj_Fact, b) +
           fCoef * CalcCoulombVir(softRsq, qi_qj_Fact, b);
  } else {
    dhdl = CalcCoulomb(distSq, qi_qj_Fact, b);
  }
  return dhdl;
}

#endif /*FF_TABULATED_H*/
