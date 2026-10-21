/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "Instrumentation.h"

#ifdef GOMC_INSTRUMENT

#include <algorithm>
#include <cmath>
#include <iostream>

#include "MoleculeKind.h"
#include "MoveConst.h"
#include "NumLib.h"
#include "StrLib.h"
#include "cbmc/TrialMol.h"

namespace instr {

namespace {
// Histogram ranges. ln(acceptance ratio) is clipped at -50: below that the
// move is orders of magnitude from threshold and the exact value no longer
// matters.
const double LNACC_LO = -50.0, LNACC_HI = 10.0;
const double LNW_LO = -60.0, LNW_HI = 20.0;
const double BETAU_LO = -60.0, BETAU_HI = 60.0;
const double WORK_LO = -20.0, WORK_HI = 60.0;

const char *const REASON_NAMES[N_REASON] = {
    "accepted",     "metropolis",   "overlap_first", "overlap_later",
    "zerow_first",  "zerow_later",  "overlap",       "lowweight_first",
    "lowweight_later", "early"};

const char *EarlyName(uint code) {
  switch (code) {
  case mv::fail_state::ROTATE_ON_SINGLE_ATOM:
    return "rotate_on_single_atom";
  case mv::fail_state::NO_MOL_OF_KIND_IN_BOX:
    return "no_mol_of_kind_in_box";
  case mv::fail_state::INVALID_MP_MOVE_DIST:
    return "invalid_mp_move_dist";
  case mv::fail_state::OUTSIDE_RELAX_RADIUS:
    return "outside_relax_radius";
  default:
    return "unknown";
  }
}

double LogAddExp(double a, double b) {
  if (a == -HUGE_VAL)
    return b;
  double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

void WriteHist(FILE *f, const Histogram &h) {
  for (ulong c : h.Counts())
    fprintf(f, " %lu", c);
}
} // namespace

Histogram::Histogram(double lo, double hi, double width)
    : lo(lo), width(width),
      counts((size_t)std::ceil((hi - lo) / width), 0) {}

void Histogram::Add(double x) {
  if (counts.empty())
    return;
  long i = (long)std::floor((x - lo) / width);
  i = std::max(0L, std::min(i, (long)counts.size() - 1));
  ++counts[i];
}

Collector &Collector::Get() {
  static Collector c;
  return c;
}

bool Collector::Key::operator<(const Key &o) const {
  if (move != o.move)
    return move < o.move;
  if (box != o.box)
    return box < o.box;
  if (kind != o.kind)
    return kind < o.kind;
  return nested < o.nested;
}

void Collector::Configure(bool enable, ulong freq,
                          const std::vector<std::string> &cats,
                          const std::string &outputName, double b,
                          const std::vector<std::string> &kinds,
                          const std::vector<double> &lambdas) {
  enabled = enable;
  if (!enabled)
    return;
  frequency = freq;
  beta = b;
  kindNames = kinds;
  lambdaVDW = lambdas;
  categories = 0;
  for (const std::string &c : cats) {
    if (c == "rejection")
      categories |= REJECTION;
    else if (c == "lnacc")
      categories |= LNACC;
    else if (c == "weights")
      categories |= WEIGHTS;
    else if (c == "nemtmc")
      categories |= NEMTMC;
    else if (c == "nemtwork")
      categories |= NEMTWORK;
    else if (c == "all")
      categories |= REJECTION | LNACC | WEIGHTS | NEMTMC;
  }
  if (cats.empty())
    categories = REJECTION | LNACC | WEIGHTS | NEMTMC;
  summaryFile = outputName + "_instrument.dat";
  if (categories & NEMTWORK) {
    std::string name = outputName + "_nemtwork.dat";
    workLog = fopen(name.c_str(), "w");
    if (workLog == nullptr) {
      std::cout << "Error: cannot open " << name << "\n";
      exit(EXIT_FAILURE);
    }
    fprintf(workLog, "# NeMTMC work per attempt, in kT\n# lambdaVDW");
    for (double l : lambdaVDW)
      fprintf(workLog, " %.6g", l);
    fprintf(workLog, "\n# W destBox kind accepted betaW bw_1 ... bw_M\n");
  }
  printf("Info: Instrumentation on, categories:%s%s%s%s%s, output every %lu "
         "steps to %s\n",
         (categories & REJECTION) ? " rejection" : "",
         (categories & LNACC) ? " lnacc" : "",
         (categories & WEIGHTS) ? " weights" : "",
         (categories & NEMTMC) ? " nemtmc" : "",
         (categories & NEMTWORK) ? " nemtwork" : "", frequency,
         summaryFile.c_str());
}

MoveStats &Collector::Stats(const Key &k) {
  auto it = moves.find(k);
  if (it != moves.end())
    return it->second;
  MoveStats &s = moves[k];
  s.lnacc = Histogram(LNACC_LO, LNACC_HI, 1.0);
  s.lnWn = Histogram(LNW_LO, LNW_HI, 1.0);
  s.lnWo = Histogram(LNW_LO, LNW_HI, 1.0);
  s.betaUn = Histogram(BETAU_LO, BETAU_HI, 1.0);
  s.betaUo = Histogram(BETAU_LO, BETAU_HI, 1.0);
  s.widomLse = -HUGE_VAL;
  s.widomLse2 = -HUGE_VAL;
  for (uint st = 0; st < N_STAGE; ++st)
    s.stage[st].bestFrac = Histogram(0.0, 1.0 + 1e-9, 0.05);
  return s;
}

void Collector::BeginMove(uint move) {
  topMove = (int)move;
  extraTrials.clear();
  windowWork.clear();
}

void Collector::AddTrial(const cbmc::TrialMol &mol) {
  if (StagesOn())
    extraTrials.push_back(&mol.InstrRecord());
}

Reason Collector::Classify(uint rejectState, bool accepted, bool overlap,
                           const cbmc::TrialMol *newMol,
                           const std::vector<const TrialRecord *> &extra) {
  if (rejectState != mv::fail_state::NO_FAIL)
    return EARLY;
  if (accepted)
    return ACCEPTED;
  // A failing first stage takes precedence: it means no cavity at all.
  const TrialRecord *fail = nullptr;
  auto consider = [&fail](const TrialRecord *r) {
    if (r->failStage < 0)
      return;
    if (fail == nullptr || (r->failStage == FIRST && fail->failStage != FIRST))
      fail = r;
  };
  if (newMol != nullptr)
    consider(&newMol->InstrRecord());
  for (const TrialRecord *r : extra)
    consider(r);
  if (fail != nullptr) {
    if (fail->failOverlap)
      return fail->failStage == FIRST ? OVERLAP_FIRST : OVERLAP_LATER;
    return fail->failStage == FIRST ? ZEROW_FIRST : ZEROW_LATER;
  }
  if (overlap)
    return OVERLAP;
  if (newMol != nullptr && newMol->GetWeight() <= SMALL_WEIGHT) {
    // bottleneck: the stage with the smallest mean trial weight
    const StageRec *worst = nullptr;
    for (const StageRec &st : newMol->InstrRecord().stages)
      if (worst == nullptr || st.lnMean < worst->lnMean)
        worst = &st;
    return (worst != nullptr && worst->stage == FIRST) ? LOWW_FIRST
                                                       : LOWW_LATER;
  }
  return METROPOLIS;
}

void Collector::MoveDone(uint move, uint box, uint kind, uint rejectState,
                         bool accepted, double lnRatio, bool overlap,
                         const cbmc::TrialMol *newMol,
                         const cbmc::TrialMol *oldMol, bool insertion) {
  if (!On(REJECTION | LNACC | WEIGHTS))
    return;
  Key key{move, box, kind, topMove >= 0 && (uint)topMove != move};
  MoveStats &s = Stats(key);
  static const std::vector<const TrialRecord *> none;
  Reason r = Classify(rejectState, accepted, overlap, newMol,
                      key.nested ? none : extraTrials);
  if (!key.nested)
    extraTrials.clear();
  ++s.reason[r];
  if (r == EARLY)
    ++s.early[rejectState];

  if (On(LNACC) && r != EARLY) {
    if (r == ACCEPTED || r == METROPOLIS)
      s.lnacc.Add(std::isnan(lnRatio) ? LNACC_LO : lnRatio);
    else
      ++s.ratioZero;
  }

  if (On(WEIGHTS) && r != EARLY && newMol != nullptr) {
    s.insertion = insertion;
    double wn = newMol->GetWeight();
    if (wn > 0.0)
      s.lnWn.Add(std::log(wn));
    else
      ++s.zeroWn;
    if (insertion) {
      // Widom: average W_new over all attempts, zero weights included
      if (wn > 0.0) {
        s.widomLse = LogAddExp(s.widomLse, std::log(wn));
        s.widomLse2 = LogAddExp(s.widomLse2, 2.0 * std::log(wn));
      }
      ++s.widomN;
    }
    const Energy &en = newMol->GetEnergy();
    s.betaUn.Add(beta * (en.inter + en.real));
    if (oldMol != nullptr) {
      double wo = oldMol->GetWeight();
      if (wo > 0.0)
        s.lnWo.Add(std::log(wo));
      const Energy &eo = oldMol->GetEnergy();
      s.betaUo.Add(beta * (eo.inter + eo.real));
    }
    const std::vector<StageRec> &ns = newMol->InstrRecord().stages;
    const std::vector<StageRec> *os =
        oldMol != nullptr ? &oldMol->InstrRecord().stages : nullptr;
    size_t np = std::max(ns.size(), os != nullptr ? os->size() : 0);
    if (s.pos.size() < np) {
      s.pos.resize(np);
      s.posStage.resize(np, 255);
    }
    for (size_t p = 0; p < np; ++p) {
      PosStats &ps = s.pos[p];
      bool hasN = p < ns.size(), hasO = os != nullptr && p < os->size();
      if (hasN) {
        s.posStage[p] = ns[p].stage;
        ++ps.nNew;
        if (ns[p].zero)
          ++ps.zeroNew;
        else {
          ps.lnNew += ns[p].lnMean;
          ps.lnBNew += ns[p].lnBonded;
          ps.best90New += ns[p].bestFrac > 0.9f;
        }
      }
      if (hasO && !(*os)[p].zero) {
        ++ps.nOld;
        ps.lnOld += (*os)[p].lnMean;
        ps.lnBOld += (*os)[p].lnBonded;
      }
      if (hasN && hasO && !ns[p].zero && !(*os)[p].zero) {
        ++ps.nPair;
        ps.diff += ns[p].lnMean - (*os)[p].lnMean;
        ps.diffB += ns[p].lnBonded - (*os)[p].lnBonded;
      }
    }
    // whole-growth split, only when every stage of both has finite weight
    if (os != nullptr && ns.size() == os->size() && !ns.empty()) {
      double dB = 0.0, dL = 0.0;
      bool finite = true;
      for (size_t p = 0; p < ns.size() && finite; ++p) {
        finite = !ns[p].zero && !(*os)[p].zero;
        dB += ns[p].lnBonded - (*os)[p].lnBonded;
        dL += ns[p].lnMean - (*os)[p].lnMean;
      }
      if (finite) {
        PairStats &ps = s.pairs;
        const double *pn = newMol->InstrRecord().pairE;
        const double *po = oldMol->InstrRecord().pairE;
        double td = 0.0;
        ++ps.n;
        for (int c = 0; c < N_PAIRCLASS; ++c) {
          double d = beta * (pn[c] - po[c]);
          ps.sN[c] += beta * pn[c];
          ps.sO[c] += beta * po[c];
          ps.sD[c] += d;
          ps.sDD[c] += d * d;
          td += d;
        }
        ps.tD += td;
        ps.tDD += td * td;
        double dE = dL - dB;
        SplitStats &sp = s.split;
        ++sp.n;
        sp.sB += dB;
        sp.sE += dE;
        sp.sBB += dB * dB;
        sp.sEE += dE * dE;
        sp.sBE += dB * dE;
      }
    }
    for (const StageRec &st : ns) {
      StageStats &ss = s.stage[st.stage];
      ++ss.n;
      ss.zero += st.zero;
      ss.allOverlap += st.allOverlap;
      if (!st.zero) {
        ss.sumLnMean += st.lnMean;
        ss.sumBestFrac += st.bestFrac;
        ss.bestOver90 += st.bestFrac > 0.9f;
        ss.bestFrac.Add(st.bestFrac);
      }
      ss.sumOverFrac += st.overFrac;
    }
  }
}

void Collector::NemtWindow(double betaDW) {
  if (On(NEMTMC | NEMTWORK))
    windowWork.push_back(betaDW);
}

void Collector::NemtDone(uint destBox, uint kind, bool accepted, double betaW,
                         uint rejectState) {
  if (!On(NEMTMC | NEMTWORK) || rejectState != mv::fail_state::NO_FAIL) {
    windowWork.clear();
    return;
  }
  if (On(NEMTMC)) {
    NemtStats &ns = nemt[std::make_pair(destBox, kind)];
    if (ns.n == 0) {
      ns.sum.assign(windowWork.size(), 0.0);
      ns.sumSq.assign(windowWork.size(), 0.0);
      ns.betaW = Histogram(WORK_LO, WORK_HI, 1.0);
    }
    ++ns.n;
    ns.accepted += accepted;
    ns.betaW.Add(betaW);
    for (size_t i = 0; i < windowWork.size() && i < ns.sum.size(); ++i) {
      ns.sum[i] += windowWork[i];
      ns.sumSq[i] += windowWork[i] * windowWork[i];
    }
  }
  if (workLog != nullptr) {
    fprintf(workLog, "W %u %u %d %.6f", destBox, kind, (int)accepted, betaW);
    for (double w : windowWork)
      fprintf(workLog, " %.5f", w);
    fprintf(workLog, "\n");
  }
  windowWork.clear();
}

void Collector::MaybeWrite(ulong step) {
  if (enabled && frequency > 0 && (step + 1) % frequency == 0)
    Write(step + 1);
}

void Collector::Write(ulong step) {
  if (!enabled)
    return;
  if (workLog != nullptr)
    fflush(workLog);
  FILE *f = fopen(summaryFile.c_str(), "w");
  if (f == nullptr) {
    std::cout << "Warning: cannot write " << summaryFile << "\n";
    return;
  }
  auto kindName = [this](uint k) {
    return k < kindNames.size() ? kindNames[k].c_str() : "?";
  };
  auto moveName = [](uint m) {
    std::string s = str::MoveTypetoStr(m);
    std::replace(s.begin(), s.end(), ' ', '_');
    return s;
  };
  fprintf(f, "# GOMC sampling instrumentation, cumulative to step %lu\n", step);
  fprintf(f, "# Rows: move box kind nested ... ; nested = 1 for moves run "
             "inside another move (NeMTMC relaxation)\n");

  if (On(REJECTION)) {
    fprintf(f, "\n[rejection]\n# move box kind nested attempts");
    for (uint r = 0; r < N_REASON; ++r)
      fprintf(f, " %s", REASON_NAMES[r]);
    fprintf(f, "\n");
    for (const auto &it : moves) {
      ulong n = 0;
      for (uint r = 0; r < N_REASON; ++r)
        n += it.second.reason[r];
      fprintf(f, "%s %u %s %d %lu", moveName(it.first.move).c_str(),
              it.first.box, kindName(it.first.kind), (int)it.first.nested, n);
      for (uint r = 0; r < N_REASON; ++r)
        fprintf(f, " %lu", it.second.reason[r]);
      fprintf(f, "\n");
    }
    fprintf(f, "\n[early]\n# move box kind nested code name count\n");
    for (const auto &it : moves)
      for (const auto &e : it.second.early)
        fprintf(f, "%s %u %s %d %u %s %lu\n", moveName(it.first.move).c_str(),
                it.first.box, kindName(it.first.kind), (int)it.first.nested,
                e.first, EarlyName(e.first), e.second);
  }

  if (On(LNACC)) {
    fprintf(f,
            "\n[lnacc]\n# ln(acceptance ratio) for accepted and Metropolis-"
            "rejected attempts; bins of width 1 from %.0f, first bin <= %.0f, "
            "last bin >= %.0f. ratio_zero = overlap/zero-weight rejections.\n"
            "# move box kind nested ratio_zero counts...\n",
            LNACC_LO, LNACC_LO + 1.0, LNACC_HI - 1.0);
    for (const auto &it : moves) {
      fprintf(f, "%s %u %s %d %lu", moveName(it.first.move).c_str(),
              it.first.box, kindName(it.first.kind), (int)it.first.nested,
              it.second.ratioZero);
      WriteHist(f, it.second.lnacc);
      fprintf(f, "\n");
    }
  }

  if (On(WEIGHTS)) {
    fprintf(f, "\n[widom]\n# insertion-type moves: ln<W_new> over all "
               "attempts (zero weights included); W includes 1/ntrials.\n"
               "# ess = (sum W)^2 / sum W^2, the effective number of "
               "insertions; the estimate is unreliable when ess is small\n"
               "# move box kind nested attempts ln_mean_W ess\n");
    for (const auto &it : moves) {
      const MoveStats &s = it.second;
      if (!s.insertion || s.widomN == 0)
        continue;
      double ess = s.widomLse2 == -HUGE_VAL
                       ? 0.0
                       : std::exp(2.0 * s.widomLse - s.widomLse2);
      fprintf(f, "%s %u %s %d %lu %.6f %.1f\n",
              moveName(it.first.move).c_str(), it.first.box,
              kindName(it.first.kind), (int)it.first.nested, s.widomN,
              s.widomLse - std::log((double)s.widomN), ess);
    }
    fprintf(f, "\n[weights]\n# histograms, bins of width 1: lnWn and lnWo "
               "from %.0f, betaU (intermolecular energy of the trial "
               "molecule, kT) from %.0f; end bins are clipped\n"
               "# move box kind nested quantity zero_count counts...\n",
            LNW_LO, BETAU_LO);
    for (const auto &it : moves) {
      const MoveStats &s = it.second;
      std::string m = moveName(it.first.move);
      const char *mn = m.c_str();
      const Histogram *h[4] = {&s.lnWn, &s.lnWo, &s.betaUn, &s.betaUo};
      const char *q[4] = {"lnWn", "lnWo", "betaUn", "betaUo"};
      for (int i = 0; i < 4; ++i) {
        ulong tot = 0;
        for (ulong c : h[i]->Counts())
          tot += c;
        if (tot == 0 && !(i == 0 && s.zeroWn))
          continue;
        fprintf(f, "%s %u %s %d %s %lu", mn, it.first.box,
                kindName(it.first.kind), (int)it.first.nested, q[i],
                i == 0 ? s.zeroWn : 0UL);
        WriteHist(f, *h[i]);
        fprintf(f, "\n");
      }
    }
    fprintf(f, "\n[stages]\n# new-configuration CBMC trial sets by stage "
               "(first = first atom / rigid placement, later = growth)\n"
               "# best_frac = largest trial weight / sum; near 1 means too "
               "few trials. Histogram of best_frac in bins of 0.05.\n"
               "# mean_ln_w = mean over nonzero sets of ln(mean trial "
               "weight); the most negative stage limits the Rosenbluth "
               "weight\n"
               "# move box kind nested stage n zero all_overlap "
               "mean_overlap_frac mean_best_frac frac_best_gt_0.9 mean_ln_w "
               "counts...\n");
    for (const auto &it : moves) {
      for (uint st = 0; st < N_STAGE; ++st) {
        const StageStats &ss = it.second.stage[st];
        if (ss.n == 0)
          continue;
        ulong nz = ss.n - ss.zero;
        fprintf(f, "%s %u %s %d %s %lu %lu %lu %.4f %.4f %.4f %.4f",
                moveName(it.first.move).c_str(), it.first.box,
                kindName(it.first.kind), (int)it.first.nested,
                st == FIRST ? "first" : "later", ss.n, ss.zero, ss.allOverlap,
                ss.sumOverFrac / ss.n, nz ? ss.sumBestFrac / nz : 0.0,
                nz ? (double)ss.bestOver90 / nz : 0.0,
                nz ? ss.sumLnMean / nz : 0.0);
        WriteHist(f, ss.bestFrac);
        fprintf(f, "\n");
      }
    }
  }

  if (On(WEIGHTS)) {
    fprintf(f, "\n[stage_position]\n# by position in the growth sequence "
               "(0 = first component built): ln(mean trial weight) of the new "
               "and old configurations; new and old share the growth order in "
               "swap moves, so mean_diff (new - old, paired) shows which "
               "growth steps lose weight\n"
               "# move box kind nested pos stage n_new zero_new "
               "frac_best_gt_0.9 mean_ln_w_new n_old mean_ln_w_old n_pair "
               "mean_diff\n");
    for (const auto &it : moves) {
      const MoveStats &s = it.second;
      for (size_t p = 0; p < s.pos.size(); ++p) {
        const PosStats &ps = s.pos[p];
        ulong nzn = ps.nNew - ps.zeroNew;
        fprintf(f,
                "%s %u %s %d %zu %s %lu %lu %.4f %.4f %lu %.4f %lu %.4f "
                "%.4f %.4f %.4f\n",
                moveName(it.first.move).c_str(), it.first.box,
                kindName(it.first.kind), (int)it.first.nested, p,
                s.posStage[p] == FIRST ? "first" : "later", ps.nNew,
                ps.zeroNew, nzn ? (double)ps.best90New / nzn : 0.0,
                nzn ? ps.lnNew / nzn : 0.0, ps.nOld,
                ps.nOld ? ps.lnOld / ps.nOld : 0.0, ps.nPair,
                ps.nPair ? ps.diff / ps.nPair : 0.0,
                nzn ? ps.lnBNew / nzn : 0.0,
                ps.nOld ? ps.lnBOld / ps.nOld : 0.0,
                ps.nPair ? ps.diffB / ps.nPair : 0.0);
      }
    }
    fprintf(f, "\n[weight_split]\n# per attempt with finite weights: "
               "d = ln(W_new/W_old) over the recorded growth stages, split "
               "into bonded dB (importance weights of branch and torsion "
               "sampling) and nonbonded dE (Boltzmann factors of the trial "
               "energies); var(d) = var(dB) + var(dE) + 2 cov\n"
               "# move box kind nested n mean_dB mean_dE var_dB var_dE "
               "cov_dBdE\n");
    for (const auto &it : moves) {
      const SplitStats &sp = it.second.split;
      if (sp.n < 2)
        continue;
      double mB = sp.sB / sp.n, mE = sp.sE / sp.n;
      fprintf(f, "%s %u %s %d %lu %.4f %.4f %.4f %.4f %.4f\n",
              moveName(it.first.move).c_str(), it.first.box,
              kindName(it.first.kind), (int)it.first.nested, sp.n, mB, mE,
              sp.sBB / sp.n - mB * mB, sp.sEE / sp.n - mE * mE,
              sp.sBE / sp.n - mB * mE);
    }
  }

  if (On(WEIGHTS)) {
    static const char *SEP[N_SEP] = {"1-5", "1-6", "1-7+"};
    static const char *TYP[N_PTYPE] = {"C-C", "C-H", "H-H", "other"};
    static const char *TERM[2] = {"LJ", "Coul"};
    fprintf(f, "\n[pair_split]\n# intramolecular 1-5+ energy formed by the "
               "growth steps that record pairs (DCLinkedHedron), by class, "
               "per attempt with finite weights; kT units; diff = new - old\n"
               "# move box kind nested sep types term n mean_new mean_old "
               "mean_diff var_diff\n");
    for (const auto &it : moves) {
      const PairStats &ps = it.second.pairs;
      if (ps.n < 2)
        continue;
      std::string mn = moveName(it.first.move);
      for (int c = 0; c < N_PAIRCLASS; ++c) {
        if (ps.sN[c] == 0.0 && ps.sO[c] == 0.0)
          continue;
        int term = c % 2, typ = (c / 2) % N_PTYPE, sep = c / (2 * N_PTYPE);
        double md = ps.sD[c] / ps.n;
        fprintf(f, "%s %u %s %d %s %s %s %lu %.4f %.4f %.4f %.4f\n",
                mn.c_str(), it.first.box, kindName(it.first.kind),
                (int)it.first.nested, SEP[sep], TYP[typ], TERM[term], ps.n,
                ps.sN[c] / ps.n, ps.sO[c] / ps.n, md,
                ps.sDD[c] / ps.n - md * md);
      }
      double mt = ps.tD / ps.n;
      fprintf(f, "%s %u %s %d all all all %lu 0 0 %.4f %.4f\n", mn.c_str(),
              it.first.box, kindName(it.first.kind), (int)it.first.nested,
              ps.n, mt, ps.tDD / ps.n - mt * mt);
    }
  }

  if (On(NEMTMC)) {
    fprintf(f, "\n[nemtmc]\n# per destination box and kind: attempts, "
               "accepted; betaW histogram in bins of 1 from %.0f\n"
               "# destBox kind attempts accepted counts...\n",
            WORK_LO);
    for (const auto &it : nemt) {
      fprintf(f, "%u %s %lu %lu", it.first.first, kindName(it.first.second),
              it.second.n, it.second.accepted);
      WriteHist(f, it.second.betaW);
      fprintf(f, "\n");
    }
    fprintf(f, "\n[nemtmc_windows]\n# window i: destination lambda goes "
               "lam[i] -> lam[i+1], source lam[M-i] -> lam[M-i-1]; work in "
               "kT\n# destBox kind window lam_lo lam_hi mean var\n");
    for (const auto &it : nemt) {
      const NemtStats &ns = it.second;
      for (size_t i = 0; i < ns.sum.size(); ++i) {
        double m = ns.sum[i] / ns.n;
        double v = ns.n > 1 ? (ns.sumSq[i] - ns.n * m * m) / (ns.n - 1) : 0.0;
        double lo = i < lambdaVDW.size() ? lambdaVDW[i] : NAN;
        double hi = i + 1 < lambdaVDW.size() ? lambdaVDW[i + 1] : NAN;
        fprintf(f, "%u %s %zu %.6g %.6g %.6f %.6f\n", it.first.first,
                kindName(it.first.second), i, lo, hi, m, v);
      }
    }
  }
  fclose(f);
}

void RecordStage(cbmc::TrialMol &mol, Stage stage, const double *w, uint n,
                 const bool *overlap) {
  if (!Collector::Get().StagesOn() || n == 0)
    return;
  double sum = 0.0, best = 0.0;
  uint nOver = 0;
  for (uint i = 0; i < n; ++i) {
    sum += w[i];
    best = std::max(best, w[i]);
    if (overlap != nullptr && overlap[i])
      ++nOver;
  }
  StageRec r;
  r.stage = (unsigned char)stage;
  r.zero = !(sum > 0.0);
  r.allOverlap = (nOver == n);
  r.bestFrac = r.zero ? 0.0f : (float)(best / sum);
  r.overFrac = (float)nOver / n;
  r.lnMean = r.zero ? -HUGE_VAL : std::log(sum / n);
  TrialRecord &rec = mol.InstrRecord();
  rec.stages.push_back(r);
  if (r.zero && rec.failStage < 0) {
    rec.failStage = stage;
    rec.failOverlap = r.allOverlap;
  }
}

void RecordStageFactor(cbmc::TrialMol &mol, Stage stage, const double *w,
                       uint n, const bool *overlap, double factor) {
  if (!Collector::Get().StagesOn() || n == 0)
    return;
  double sum = 0.0, best = 0.0;
  uint nOver = 0;
  for (uint i = 0; i < n; ++i) {
    sum += w[i];
    best = std::max(best, w[i]);
    if (overlap[i])
      ++nOver;
  }
  StageRec r;
  r.stage = (unsigned char)stage;
  r.zero = !(factor > 0.0);
  r.allOverlap = (nOver == n);
  r.bestFrac = sum > 0.0 ? (float)(best / sum) : 0.0f;
  r.overFrac = (float)nOver / n;
  r.lnMean = r.zero ? -HUGE_VAL : std::log(factor);
  TrialRecord &rec = mol.InstrRecord();
  rec.stages.push_back(r);
  if (r.zero && rec.failStage < 0) {
    rec.failStage = stage;
    rec.failOverlap = r.allOverlap;
  }
}

const Collector::Topo &Collector::Topology(const cbmc::TrialMol &mol) {
  const MoleculeKind &kind = mol.GetKind();
  auto it = topo.find(&kind);
  if (it != topo.end())
    return it->second;
  Topo &t = topo[&kind];
  t.n = kind.NumAtoms();
  t.dist.assign(t.n * t.n, 255);
  std::vector<std::vector<uint>> adj(t.n);
  for (uint b = 0; b < kind.bondList.count; ++b) {
    adj[kind.bondList.part1[b]].push_back(kind.bondList.part2[b]);
    adj[kind.bondList.part2[b]].push_back(kind.bondList.part1[b]);
  }
  for (uint s = 0; s < t.n; ++s) {
    std::vector<uint> queue(1, s);
    t.dist[s * t.n + s] = 0;
    for (size_t q = 0; q < queue.size(); ++q) {
      uint a = queue[q];
      for (uint nb : adj[a]) {
        if (t.dist[s * t.n + nb] == 255) {
          t.dist[s * t.n + nb] = t.dist[s * t.n + a] + 1;
          queue.push_back(nb);
        }
      }
    }
  }
  t.elem.resize(t.n);
  for (uint a = 0; a < t.n; ++a) {
    char e = kind.atomTypeNames[a].empty() ? 'X' : kind.atomTypeNames[a][0];
    e = (char)toupper(e);
    t.elem[a] = (e == 'C' || e == 'H') ? e : 'X';
  }
  return t;
}

void RecordPairs(cbmc::TrialMol &mol, uint atom,
                 const std::vector<PairE> &pairs) {
  Collector &col = Collector::Get();
  if (!col.On(WEIGHTS))
    return;
  const Collector::Topo &t = col.Topology(mol);
  double *pe = mol.InstrRecord().pairE;
  for (const PairE &p : pairs) {
    uint d = t.dist[atom * t.n + p.partner];
    if (d < 4)
      continue; // not a 1-5+ pair
    int sep = d == 4 ? 0 : (d == 5 ? 1 : 2);
    char a = t.elem[atom], b = t.elem[p.partner];
    int typ = 3;
    if (a == 'C' && b == 'C')
      typ = 0;
    else if ((a == 'C' && b == 'H') || (a == 'H' && b == 'C'))
      typ = 1;
    else if (a == 'H' && b == 'H')
      typ = 2;
    int c = (sep * N_PTYPE + typ) * 2;
    pe[c] += p.lj;
    pe[c + 1] += p.coul;
  }
}

void RecordBondedPart(cbmc::TrialMol &mol, double lnBonded) {
  if (!Collector::Get().On(WEIGHTS))
    return;
  std::vector<StageRec> &st = mol.InstrRecord().stages;
  if (!st.empty())
    st.back().lnBonded = lnBonded;
}

void RecordOldStage(cbmc::TrialMol &mol, Stage stage, double meanW) {
  if (!Collector::Get().On(WEIGHTS))
    return;
  StageRec r;
  r.stage = (unsigned char)stage;
  r.zero = !(meanW > 0.0);
  r.allOverlap = false;
  r.bestFrac = 0.0f;
  r.overFrac = 0.0f;
  r.lnMean = r.zero ? -HUGE_VAL : std::log(meanW);
  mol.InstrRecord().stages.push_back(r);
}

} // namespace instr

#endif /*GOMC_INSTRUMENT*/
