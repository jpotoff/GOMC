/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#ifndef INSTRUMENTATION_H
#define INSTRUMENTATION_H

// Sampling diagnostics for CBMC and transfer moves.
//
// Compiled in only when GOMC_INSTRUMENT is defined (CMake -DGOMC_INSTRUMENT=on
// or metamake -i). Without it every GOMC_INSTR(...) hook expands to nothing,
// so production builds carry no cost. With it, the "Instrument" config
// keyword selects what is collected at runtime:
//
//   Instrument  true  [frequency]  [categories...]
//
// Categories: rejection, lnacc, weights, nemtmc, nemtwork, all (default: all
// except nemtwork). Results go to <OutputName>_instrument.dat, rewritten with
// cumulative totals every `frequency` steps and at the end of the run;
// nemtwork streams one line per NeMTMC attempt to <OutputName>_nemtwork.dat.
// scripts/instrument_report.py interprets the summary file.

#ifdef GOMC_INSTRUMENT
#define GOMC_INSTR(x) x
#else
#define GOMC_INSTR(x)
#endif

#ifdef GOMC_INSTRUMENT

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "BasicTypes.h"

namespace cbmc {
class TrialMol;
}

namespace instr {

// Growth stage of a CBMC trial set. FIRST is the placement of the first atom
// (DCSingle) or of a rigid molecule's position and orientation (DCRotateCOM):
// when it fails there is no cavity to put the molecule in. LATER is every
// subsequent growth step: when it fails the molecule has a cavity but no
// conformation that fits it.
enum Stage { FIRST = 0, LATER = 1, N_STAGE = 2 };

enum Category {
  REJECTION = 1 << 0, // rejection reason and failing stage
  LNACC = 1 << 1,     // histogram of ln(acceptance ratio)
  WEIGHTS = 1 << 2,   // Rosenbluth weights, insertion energies, trial stats
  NEMTMC = 1 << 3,    // NeMTMC per-window work statistics
  NEMTWORK = 1 << 4   // NeMTMC per-attempt work log (large)
};

// Outcome of one move attempt.
enum Reason {
  ACCEPTED = 0,
  METROPOLIS,    // finite acceptance ratio, lost the random draw
  OVERLAP_FIRST, // every first-stage trial was inside RcutLow
  OVERLAP_LATER, // every trial of a later growth stage was inside RcutLow
  ZEROW_FIRST,   // first-stage trial weights all zero, not all overlaps
  ZEROW_LATER,   // later-stage trial weights all zero, not all overlaps
  OVERLAP,       // move-level overlap with no failing stage (rigid moves,
                 // MEMC, or an overlapping trial picked from nonzero weights)
  LOWW_FIRST,    // no stage failed but the total weight fell below
  LOWW_LATER,    // SMALL_WEIGHT; attributed to the stage with the smallest
                 // mean trial weight (the bottleneck)
  EARLY,         // move rejected before trial generation (rejectState)
  N_REASON
};

// One CBMC trial set (one call to PickWeighted on the new configuration).
struct StageRec {
  unsigned char stage;
  bool zero;       // all trial weights are zero
  bool allOverlap; // every trial overlapped
  float bestFrac;  // largest trial weight / sum of weights
  float overFrac;  // fraction of trials that overlapped
  double lnMean;   // ln(mean trial weight), -HUGE_VAL if all zero
};

// Per-trial-molecule record, carried by cbmc::TrialMol.
struct TrialRecord {
  int failStage = -1; // first stage with all weights zero, -1 if none
  bool failOverlap = false;
  std::vector<StageRec> stages;
};

// Fixed-width histogram; values outside [lo, hi) go into the end bins.
class Histogram {
public:
  Histogram() : lo(0.0), width(1.0) {}
  Histogram(double lo, double hi, double width);
  void Add(double x);
  double Lo() const { return lo; }
  double Width() const { return width; }
  const std::vector<ulong> &Counts() const { return counts; }

private:
  double lo, width;
  std::vector<ulong> counts;
};

struct StageStats {
  ulong n = 0, zero = 0, allOverlap = 0, bestOver90 = 0;
  double sumBestFrac = 0.0, sumOverFrac = 0.0, sumLnMean = 0.0;
  Histogram bestFrac;
};

// Per position in the growth sequence. For swaps the old and new
// configurations are grown in the same order, so position p of each is the
// same growth step and their ln(mean weight) can be compared.
struct PosStats {
  ulong nNew = 0, zeroNew = 0, best90New = 0, nOld = 0, nPair = 0;
  double lnNew = 0.0, lnOld = 0.0, diff = 0.0;
};

struct MoveStats {
  ulong reason[N_REASON] = {};
  std::map<uint, ulong> early; // rejectState code -> count
  Histogram lnacc;
  ulong ratioZero = 0; // structural rejections (overlap / zero weight)
  Histogram lnWn, lnWo, betaUn, betaUo;
  ulong zeroWn = 0;
  bool insertion = false;
  double widomLse = 0.0;  // log(sum of W_new) over insertion attempts
  double widomLse2 = 0.0; // log(sum of W_new^2), for the effective sample size
  ulong widomN = 0;
  StageStats stage[N_STAGE];
  std::vector<PosStats> pos;
  std::vector<unsigned char> posStage; // stage type seen at each position
};

struct NemtStats {
  ulong n = 0, accepted = 0;
  std::vector<double> sum, sumSq; // per-window beta*dW
  Histogram betaW;
};

class Collector {
public:
  static Collector &Get();

  void Configure(bool enable, ulong frequency,
                 const std::vector<std::string> &categories,
                 const std::string &outputName, double beta,
                 const std::vector<std::string> &kindNames,
                 const std::vector<double> &lambdaVDW);
  bool On(unsigned cats) const { return enabled && (categories & cats); }
  // Stage records are only needed for rejection and weight diagnostics.
  bool StagesOn() const { return On(REJECTION | WEIGHTS); }

  // Called by System::RunMove around each top-level move. Moves invoked from
  // inside another move (NeMTMC relaxation) are recorded as nested.
  void BeginMove(uint move);
  // Register an extra trial molecule for the current attempt (MEMC builds
  // several); its stage failures count toward the rejection reason.
  void AddTrial(const cbmc::TrialMol &mol);

  // Record one move attempt. lnRatio is ln of the acceptance ratio; it is
  // ignored for structural and early rejections. newMol/oldMol may be null
  // for moves without CBMC trial molecules. insertion marks moves whose new
  // weight is a Widom insertion weight (swap, intra-swap, targeted swap).
  void MoveDone(uint move, uint box, uint kind, uint rejectState,
                bool accepted, double lnRatio, bool overlap,
                const cbmc::TrialMol *newMol, const cbmc::TrialMol *oldMol,
                bool insertion);

  // NeMTMC: work increment of one lambda window (in kT), then the attempt.
  void NemtWindow(double betaDW);
  void NemtDone(uint destBox, uint kind, bool accepted, double betaW,
                uint rejectState);

  void MaybeWrite(ulong step);
  void Write(ulong step);

private:
  Collector() = default;
  struct Key {
    uint move, box, kind;
    bool nested;
    bool operator<(const Key &o) const;
  };
  MoveStats &Stats(const Key &k);
  static Reason Classify(uint rejectState, bool accepted, bool overlap,
                         const cbmc::TrialMol *newMol,
                         const std::vector<const TrialRecord *> &extra);

  bool enabled = false;
  unsigned categories = 0;
  ulong frequency = 0;
  double beta = 0.0;
  std::string summaryFile;
  std::vector<std::string> kindNames;
  std::vector<double> lambdaVDW;
  FILE *workLog = nullptr;
  int topMove = -1;
  std::vector<const TrialRecord *> extraTrials;
  std::vector<double> windowWork;
  std::map<Key, MoveStats> moves;
  std::map<std::pair<uint, uint>, NemtStats> nemt; // (destBox, kind)
};

// Record one CBMC trial set of the new configuration on its trial molecule.
// w: trial weights, overlap: per-trial overlap flags (may be null).
void RecordStage(cbmc::TrialMol &mol, Stage stage, const double *w, uint n,
                 const bool *overlap);
// Record one stage of the old configuration (mean trial weight only; the old
// configuration cannot fail, since trial 0 is the actual molecule).
void RecordOldStage(cbmc::TrialMol &mol, Stage stage, double meanW);

} // namespace instr

#endif /*GOMC_INSTRUMENT*/
#endif /*INSTRUMENTATION_H*/
