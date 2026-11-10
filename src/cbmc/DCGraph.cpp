/******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) Copyright (C) GOMC Group
A copy of the MIT License can be found in License.txt with this program or at
<https://opensource.org/licenses/MIT>.
******************************************************************************/
#include "DCGraph.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>

#include "DCCrankShaftAng.h"
#include "DCCrankShaftDih.h"
#include "DCFreeHedron.h"
#include "DCFreeHedronSeed.h"
#include "DCLinkedHedron.h"
#include "DCRotateCOM.h"
#include "RecoilGrowth.h"
#include "SMCGrowth.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cbmc {

namespace {
// Recoil growth provider (RecoilGrowth.h) over the edge steps of a DCGraph
// growth. Place/Unplace put a trial's atoms into the trial molecule and
// restore what was there before, so the old configuration's coordinates
// survive the tentative placements made while weighing it.
class EdgeSteps {
public:
  typedef DCLinkedHedron::StepTrial Trial;
  EdgeSteps(TrialMol &mol, uint molIndex,
            const std::vector<DCLinkedHedron *> &steps, bool old)
      : mol(mol), molIndex(molIndex), steps(steps), old(old) {}

  uint NumSteps() const { return steps.size(); }
  void Generate(uint s, uint n, Trial *out) {
    steps[s]->GenerateTrials(mol, molIndex, n, old, out);
  }
  void Actual(uint s, Trial &out) { steps[s]->ActualTrial(mol, molIndex, out); }
  void Place(uint s, const Trial &t) {
    const DCLinkedHedron *c = steps[s];
    undo.push_back(Saved());
    Saved &sv = undo.back();
    for (uint b = 0; b < c->NumBonded(); ++b) {
      uint a = c->BondedAtom(b);
      sv.pos[b] = mol.AtomPosition(a);
      sv.built[b] = mol.AtomExists(a);
      if (t.actual)
        mol.ConfirmOldAtom(a); // keep the stored coordinates bit for bit
      else
        mol.AddAtom(a, t.pos[b]);
    }
  }
  void Unplace(uint s) {
    const DCLinkedHedron *c = steps[s];
    const Saved &sv = undo.back();
    for (uint b = 0; b < c->NumBonded(); ++b) {
      uint a = c->BondedAtom(b);
      mol.SetAtomCoords(a, sv.pos[b]);
      if (!sv.built[b])
        mol.RemoveAtom(a);
    }
    undo.pop_back();
  }

private:
  struct Saved {
    XYZ pos[MAX_BONDS];
    bool built[MAX_BONDS];
  };
  TrialMol &mol;
  uint molIndex;
  const std::vector<DCLinkedHedron *> &steps;
  bool old;
  std::vector<Saved> undo;
};

#ifdef GOMC_INSTRUMENT
// Record each recoil growth step as one stage; its Rosenbluth factor is
// m_i w_i / (k p_i) rather than the mean trial weight of standard CBMC.
template <class RG>
void RecordRecoilStages(TrialMol &mol, const RG &rg, bool old) {
  if (!instr::Collector::Get().StagesOn())
    return;
  for (uint i = 0; i < rg.StepsDone(); ++i) {
    if (old) {
      instr::RecordOldStage(mol, instr::LATER, rg.Factor(i));
      continue;
    }
    const auto &tr = rg.StepTrials(i);
    std::vector<double> w(tr.size());
    std::unique_ptr<bool[]> ov(new bool[tr.size()]);
    for (size_t j = 0; j < tr.size(); ++j) {
      w[j] = tr[j].w;
      ov[j] = tr[j].overlap;
    }
    instr::RecordStageFactor(mol, instr::LATER, w.data(), tr.size(), ov.get(),
                             rg.Factor(i));
  }
}
#endif
} // namespace

struct DCGraph::RecoilState {
  RecoilState(uint k, uint l) : rg(k, l) {}
  RecoilGrowth<EdgeSteps, PRNG> rg;
  std::vector<DCLinkedHedron *> seq;
};
DCGraph::DCGraph(System &sys, const Forcefield &ff, const MoleculeKind &kind,
                 const Setup &set)
    : data(sys, ff, set) {
  Init(sys, ff, kind, set, false);
}

DCGraph::DCGraph(System &sys, const Forcefield &ff, const MoleculeKind &kind,
                 const Setup &set, PRNG *rng)
    : data(sys, ff, set, rng) {
  Init(sys, ff, kind, set, true);
}

void DCGraph::Init(System &sys, const Forcefield &ff, const MoleculeKind &kind,
                   const Setup &set, bool worker) {
  using namespace mol_setup;
  MolMap::const_iterator it = set.mol.kindMap.find(kind.uniqueName);
  assert(it != set.mol.kindMap.end());
  const MolKind setupKind = it->second;

  idExchange = new DCRotateCOM(&data, setupKind);
  // init the coordinate
  coords.Init(setupKind.atoms.size());

  std::vector<uint> atomToNode(setupKind.atoms.size(), 0);
  std::vector<uint> bondCount(setupKind.atoms.size(), 0);
  // Count the number of bonds for each atom
  for (uint b = 0; b < setupKind.bonds.size(); ++b) {
    const Bond &bond = setupKind.bonds[b];
    ++bondCount[bond.a0];
    ++bondCount[bond.a1];
  }

  // Find the node (number of bound > 1)
  // Construct the starting node (DCFreeHedron)
  // Construct the Linking node (DCLinkHedron)
  for (uint atom = 0; atom < setupKind.atoms.size(); ++atom) {
    if (bondCount[atom] < 2) {
      atomToNode[atom] = -1;
    } else {
      // Get the information of other Atoms that are bonded to the atom
      std::vector<Bond> bonds = AtomBonds(setupKind, atom);
      atomToNode[atom] = nodes.size();
      // Add atom to the node list and initialize it with DCFreeHedron, atom and
      // the first partner of the atom
      nodes.push_back(Node());
      Node &node = nodes.back();
      // Atoms bonded to atom will be build from focus (atom) in random loc.
      node.starting = new DCFreeHedron(&data, setupKind, atom, bonds[0].a1);
      // Atoms bonded to atom will be build from focus (atom) in specified loc.
      node.restarting =
          new DCFreeHedronSeed(&data, setupKind, atom, bonds[0].a1);
      // set the atom index of the node
      node.atomIndex = atom;
      // Loop through all the bonds
      for (uint i = 0; i < bonds.size(); ++i) {
        uint partner = bonds[i].a1;
        // Store partner index for each node
        node.partnerIndex.push_back(partner);
        if (bondCount[partner] == 1) {
          continue;
        }
        // Add partner to the edge list of node and initialize it with partner
        // and the atom in DCLinkedHedron
        // Atoms will be build from prev(atom) to focus (partner)
        Edge e =
            Edge(partner, new DCLinkedHedron(&data, setupKind, partner, atom));
        node.edges.push_back(e);
      }
    }
  }

  // reassign destination values from atom indices to node indices
  for (uint i = 0; i < nodes.size(); ++i) {
    for (uint j = 0; j < nodes[i].edges.size(); ++j) {
      int &dest = nodes[i].edges[j].destination;
      dest = atomToNode[dest];
      assert(dest != -1);
    }
  }

  // Link each linked hedron to the steps that grow from the atoms it places,
  // for the one-step look-ahead (CBMC_LookAhead)
  for (uint i = 0; i < nodes.size(); ++i) {
    for (uint j = 0; j < nodes[i].edges.size(); ++j) {
      Edge &e = nodes[i].edges[j];
      std::vector<DCLinkedHedron *> kids;
      for (Edge &f : nodes[e.destination].edges)
        if (f.destination != (int)i)
          kids.push_back(static_cast<DCLinkedHedron *>(f.component));
      static_cast<DCLinkedHedron *>(e.component)->SetChildren(kids);
    }
  }

  InitCrankShaft(setupKind);
  smcCapable = true;
  for (const Node &nd : nodes)
    for (const Edge &e : nd.edges)
      if (dynamic_cast<DCLinkedHedron *>(e.component) == nullptr)
        smcCapable = false;
  if (data.smc && !smcCapable)
    std::cout << "Warning: CBMC_SMC is not supported for molecule kind "
              << kind.name << "; it is grown by standard CBMC.\n";
  for (const Node &nd : nodes)
    for (const Edge &e : nd.edges) {
      edgeFlat[e.component] = flatEdges.size();
      flatEdges.push_back(dynamic_cast<DCLinkedHedron *>(e.component));
    }
#if defined(_OPENMP) && !defined(GOMC_INSTRUMENT)
  // One worker graph per extra thread, so SMC particles can be extended in
  // parallel. Each is built from the same setup, so its nodes and edges match
  // this graph's one for one (flatEdges), with its own scratch and generator.
  // Instrumented builds stay serial: the collector is not thread safe.
  if (data.smc && smcCapable && !worker) {
    const int T = omp_get_max_threads();
    for (int t = 1; t < T; ++t) {
      SMCWorker w;
      w.prng = new PRNG(sys.molLookupRef);
      w.gen = new MTRand(1u); // reseeded from data.prng at every growth
      w.prng->Init(w.gen);
      w.graph = new DCGraph(sys, ff, kind, set, w.prng);
      smcWorkers.push_back(w);
    }
  }
#else
  (void)sys;
  (void)ff;
  (void)set;
  (void)worker;
#endif
  recoil = data.recoil
               ? new RecoilState(data.recoilTrials, data.recoilLength)
               : nullptr;
}

void DCGraph::InitCrankShaft(const mol_setup::MolKind &kind) {
  using namespace mol_setup;
  std::vector<uint> bondCount(kind.atoms.size(), 0);
  std::vector<Bond> allBonds = BondsAll(kind);
  // Count the number of bonds for each atom
  for (uint b = 0; b < allBonds.size(); ++b) {
    ++bondCount[allBonds[b].a0];
    ++bondCount[allBonds[b].a1];
  }

  // Start with atoms that form dihedral
  std::vector<Dihedral> dihs = DihsAll(kind);
  for (uint d = 0; d < dihs.size(); d++) {
    // find the last atom index in the dihedral
    uint a0 = dihs[d].a0;
    uint a1 = dihs[d].a1;
    uint a2 = dihs[d].a2;
    uint a3 = dihs[d].a3;
    // ignore single bonded atoms
    if (bondCount[a0] == 1 && bondCount[a3] == 1) {
      continue;
    }

    bool fixAngle = false;
    // Find all the angle that forms x-a0-a1
    std::vector<Angle> angle = AtomMidEndAngles(kind, a0, a1);
    // Find all the angle that forms a2-a3-x
    std::vector<Angle> tempAng = AtomMidEndAngles(kind, a3, a2);
    // merge all the angle
    angle.insert(angle.end(), tempAng.begin(), tempAng.end());
    // Check to see if any of these angles are fixed or not.
    for (uint a = 0; a < angle.size(); a++) {
      if (data.ff.angles->AngleFixed(angle[a].kind)) {
        fixAngle = true;
      }
    }

    // If there was no fix angles, we create DCCrankShaftDih
    if (!fixAngle) {
      shaftNodes.push_back(new DCCrankShaftDih(&data, kind, a0, a1, a2, a3));
    }
  }

  // Continue with the atoms that form angles.
  std::vector<Angle> angles = AngsAll(kind);
  for (uint a = 0; a < angles.size(); a++) {
    // find the last atom index in the angle
    uint a0 = angles[a].a0;
    uint a1 = angles[a].a1;
    uint a2 = angles[a].a2;
    // ignore single bonded atoms
    if (bondCount[a0] == 1 && bondCount[a2] == 1) {
      continue;
    }

    bool fixAngle = false;
    // Find all the angle that forms x-a0-a1
    std::vector<Angle> angle = AtomMidEndAngles(kind, a0, a1);
    // Find all the angle that forms a1-a2-x
    std::vector<Angle> tempAng = AtomMidEndAngles(kind, a2, a1);
    // merge all the angle
    angle.insert(angle.end(), tempAng.begin(), tempAng.end());
    // Check to see if any of these angles are fixed or not.
    for (uint i = 0; i < angle.size(); i++) {
      if (data.ff.angles->AngleFixed(angle[i].kind)) {
        fixAngle = true;
      }
    }

    // If there was no fix angles, we create DCCrankShaftAngle
    if (!fixAngle) {
      shaftNodes.push_back(new DCCrankShaftAng(&data, kind, a0, a1, a2));
    }
  }

  hasCrankShaft = (shaftNodes.size() != 0);
}

void DCGraph::CrankShaft(TrialMol &oldMol, TrialMol &newMol, uint molIndex) {
  if (!hasCrankShaft) {
    // No crank shaft move for molecule with less than 4 nodes.
    // Instead we perform Regrowth move within the same box
    Regrowth(oldMol, newMol, molIndex);
  } else {
    // Set coords to coordinate of actual molecule, it will be modified
    // No need to unwrap because box is same.
    oldMol.GetCoords().CopyRange(coords, 0, 0, coords.Count());
    newMol.SetCoords(coords, 0);
    // Pick a random node pair
    uint pick = data.prng.randIntExc(shaftNodes.size());
    shaftNodes[pick]->PrepareNew(newMol, molIndex);
    shaftNodes[pick]->BuildNew(newMol, molIndex);
    shaftNodes[pick]->PrepareOld(oldMol, molIndex);
    shaftNodes[pick]->BuildOld(oldMol, molIndex);
  }
}

void DCGraph::Build(TrialMol &oldMol, TrialMol &newMol, uint molIndex) {
  if (data.smc && smcCapable && BuildSMC(oldMol, newMol, molIndex))
    return;
  // Randomly pick a node to call DCFreeHedron on it
  uint current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareNew(newMol, molIndex);
  comp->BuildNew(newMol, molIndex);
  comp->PrepareOld(oldMol, molIndex);
  comp->BuildOld(oldMol, molIndex);
  if (recoil != nullptr) {
    BuildEdgesRecoil(oldMol, newMol, molIndex, current);
    return;
  }
  // Advance along edges, building as we go
  BuildEdges(oldMol, newMol, molIndex, current);
}

void DCGraph::BuildEdgesRecoil(TrialMol &oldMol, TrialMol &newMol,
                               uint molIndex, uint current) {
  // the same random fringe walk as BuildEdges, collecting the steps; every
  // edge is a DCLinkedHedron (see the constructor)
  std::vector<DCLinkedHedron *> &seq = recoil->seq;
  WalkEdges(current, seq);

  RecoilGrowth<EdgeSteps, PRNG> &rg = recoil->rg;
  GOMC_INSTR(rg.KeepSteps(instr::Collector::Get().StagesOn());)
  EdgeSteps grow(newMol, molIndex, seq, false);
  double wn = rg.GrowNew(grow, data.prng);
  GOMC_INSTR(RecordRecoilStages(newMol, rg, false);)
  if (wn > 0.0) {
    for (uint i = 0; i < seq.size(); ++i)
      seq[i]->Commit(newMol, rg.Chosen(i));
  }
  newMol.MultWeight(wn);

  // the old configuration's bond lengths, before any tentative placement
  for (uint i = 0; i < seq.size(); ++i)
    seq[i]->CacheOldBonds(oldMol);
  EdgeSteps weigh(oldMol, molIndex, seq, true);
  double wo = rg.WeighOld(weigh, data.prng);
  GOMC_INSTR(RecordRecoilStages(oldMol, rg, true);)
  for (uint i = 0; i < seq.size(); ++i)
    seq[i]->Commit(oldMol, rg.Chosen(i));
  oldMol.MultWeight(wo);
}

void DCGraph::WalkEdges(uint current, std::vector<DCLinkedHedron *> &seq) {
  seq.clear();
  fringe = nodes[current].edges;
  while (!fringe.empty()) {
    uint pick = data.prng.randIntExc(fringe.size());
    seq.push_back(static_cast<DCLinkedHedron *>(fringe[pick].component));
    current = fringe[pick].destination;
    fringe[pick] = fringe.back();
    fringe.pop_back();
    visited[current] = true;
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

namespace {
// Shared state of one SMC growth (DCGraph::BuildSMC and ProbeSMC)
struct GraphGrowth {
  DCData *data;
  DCComponent *start;
  const std::vector<DCLinkedHedron *> *seq;
  uint molIndex, k;
  std::vector<DCLinkedHedron::StepTrial> buf;
};

// One SMC particle: a partial molecule grown by standard CBMC with k trials
// per step. Step 0 is the start node's DCFreeHedron (first atom and its
// bonded atoms, with nLJTrialsFirst = nLJTrialsNth = k); step s > 0 is edge
// step s - 1.
//
// With several threads (SMCGrowth::SetThreads), g holds one GraphGrowth per
// thread, each on its own graph (the main one or a worker), and a particle
// extends with the calling thread's.
class GraphParticle {
public:
  GraphParticle(std::vector<GraphGrowth> &g, const TrialMol &m)
      : g(&g), mol(m) {
    mol.UpdateOverlap(m.HasOverlap());
  }
  GraphParticle(const GraphParticle &o) : g(o.g), mol(o.mol) {
    mol.UpdateOverlap(o.mol.HasOverlap());
  }
  GraphParticle &operator=(const GraphParticle &o) {
    g = o.g;
    mol = o.mol;
    mol.UpdateOverlap(o.mol.HasOverlap());
    return *this;
  }

  uint NumSteps() const { return (*g)[0].seq->size() + 1; }
  const TrialMol &Mol() const { return mol; }
  double Extend(uint step, bool reference) {
#ifdef _OPENMP
    GraphGrowth *gg = &(*g)[omp_get_thread_num()];
#else
    GraphGrowth *gg = &(*g)[0];
#endif
    return Extend(*gg, step, reference);
  }

private:
  double Extend(GraphGrowth &gr, uint step, bool reference) {
    GraphGrowth *g = &gr; // shadows the member below
    if (step == 0) {
      mol.SetWeight(1.0);
      if (reference) {
        g->start->PrepareOld(mol, g->molIndex);
        g->start->BuildOld(mol, g->molIndex);
      } else {
        g->start->PrepareNew(mol, g->molIndex);
        g->start->BuildNew(mol, g->molIndex);
      }
      double W = mol.GetWeight();
      mol.SetWeight(1.0);
      return W;
    }
    DCLinkedHedron *h = (*g->seq)[step - 1];
    const uint k = g->k;
    DCLinkedHedron::StepTrial *t = g->buf.data();
    if (reference) {
      h->ActualTrial(mol, g->molIndex, t[0]);
      if (k > 1)
        h->GenerateTrials(mol, g->molIndex, k - 1, true, t + 1);
    } else {
      h->GenerateTrials(mol, g->molIndex, k, false, t);
    }
    double sum = 0.0;
    for (uint j = 0; j < k; ++j)
      sum += t[j].w;
    if (!(sum > 0.0))
      return 0.0;
    uint pick = 0;
    if (!reference) {
      double x = g->data->prng() * sum, c = 0.0;
      for (pick = 0; pick < k - 1; ++pick) {
        c += t[pick].w;
        if (x < c)
          break;
      }
      while (t[pick].w <= 0.0 && pick > 0) // guard round-off
        --pick;
    }
    const DCLinkedHedron::StepTrial &tr = t[pick];
    for (uint b = 0; b < h->NumBonded(); ++b) {
      uint a = h->BondedAtom(b);
      if (tr.actual)
        mol.ConfirmOldAtom(a);
      else
        mol.AddAtom(a, tr.pos[b]);
    }
    h->Commit(mol, tr);
    return sum / k;
  }

  std::vector<GraphGrowth> *g;
  TrialMol mol;
};

// Copy a grown particle's molecule into a move's TrialMol (whose box and
// kind it shares); TrialMol assignment drops the overlap flag
void TakeParticle(TrialMol &dst, const TrialMol &src) {
  dst = src;
  dst.UpdateOverlap(src.HasOverlap());
}
} // namespace

// Population (SMC) growth for Build (CBMC_SMC; see SMCGrowth.h). N particles
// of k trials per step (CBMC_First = CBMC_Nth = k for the start node) grow
// the new molecule, and one is chosen; conditional SMC with the same settings
// weighs the old one. Each molecule's weight is its Z_hat, so the usual
// acceptance W_new / W_old applies unchanged. The edge order is shared, as in
// Build. The look-ahead (CBMC_LookAhead) is not used.
bool DCGraph::BuildSMC(TrialMol &oldMol, TrialMol &newMol, uint molIndex) {
  if (newMol.HasCav() || newMol.COMFix() || newMol.RotateBB() ||
      oldMol.HasCav() || oldMol.COMFix() || oldMol.RotateBB())
    return false;
  uint current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  visited[current] = true;
  WalkEdges(current, smcSeq);

  // Thread t grows with graph t: this one for t = 0, worker t - 1 otherwise,
  // on the same start node and edge order. Workers draw from their own
  // generators, reseeded here from data.prng, so a run is reproducible for a
  // fixed thread count (and with one thread uses no extra random numbers).
  const uint T = 1 + smcWorkers.size();
  const uint k = data.smcTrials;
  smcSeqT.resize(T);
  std::vector<GraphGrowth> ctx(T);
  for (uint t = 0; t < T; ++t) {
    DCGraph &gr = t == 0 ? *this : *smcWorkers[t - 1].graph;
    std::vector<DCLinkedHedron *> &seq = smcSeqT[t];
    seq.resize(smcSeq.size());
    for (size_t i = 0; i < smcSeq.size(); ++i)
      seq[i] = gr.flatEdges[edgeFlat[smcSeq[i]]];
    for (DCLinkedHedron *h : seq)
      h->CacheOldBonds(oldMol);
    if (t > 0)
      smcWorkers[t - 1].gen->seed(data.prng.randInt(0xFFFFFFFFu));
    ctx[t].data = &gr.data;
    ctx[t].start = gr.nodes[current].starting;
    ctx[t].seq = &seq;
    ctx[t].molIndex = molIndex;
    ctx[t].k = k;
    ctx[t].buf.resize(k);
  }

  const uint first0 = data.nLJTrialsFirst, nth0 = data.nLJTrialsNth;
  for (GraphGrowth &c : ctx)
    c.data->nLJTrialsFirst = c.data->nLJTrialsNth = k;
  // particles have few trials per step; the candidate list still pays
  const uint minTrials0 = data.calc.SetLocalMinTrials(1);
  SMCGrowth<GraphParticle, PRNG> smc(data.smcParticles, 2.0);
  smc.SetThreads(T);

  if (T > 1)
    data.calc.PrepareParallelParticleInter(newMol.GetBox());
  double ln = smc.GrowNew(GraphParticle(ctx, newMol), data.prng);
  if (ln > -std::numeric_limits<double>::infinity()) {
    TakeParticle(newMol, smc.Chosen().Mol());
    newMol.SetWeight(std::exp(ln));
  } else {
    newMol.SetWeight(0.0); // rejected by the move's weight check
  }
  if (T > 1)
    data.calc.PrepareParallelParticleInter(oldMol.GetBox());
  double lo = smc.WeighOld(GraphParticle(ctx, oldMol), data.prng);
  TakeParticle(oldMol, smc.Chosen().Mol());
  oldMol.SetWeight(std::exp(lo));

  for (GraphGrowth &c : ctx) {
    c.data->nLJTrialsFirst = first0;
    c.data->nLJTrialsNth = nth0;
  }
  data.calc.SetLocalMinTrials(minTrials0);
  return true;
}

#ifdef GOMC_INSTRUMENT

// SMC probe: for each (N, k) setting, out receives ln Z_hat_new,
// ln Z_hat_old and the wall time in seconds of the pair. Grow a new molecule into newMol's box
// by SMC with N particles of k trials each (resampling every step) and weigh
// oldMol's configuration by conditional SMC with the same settings. The
// caller removes oldMol's molecule from the cell list first, so both see the
// same environment and the true Rosenbluth normalisation Z is the same for
// both: ln Z_hat_new - ln Z_hat_old is pure estimator error. N = 1 is
// standard CBMC with k trials at every step (CBMC_First = CBMC_Nth = k).
// All settings use the same start node and edge order. Nothing is committed
// to the system; trial counts are restored afterwards.
bool DCGraph::ProbeSMC(TrialMol &oldMol, TrialMol &newMol, uint molIndex,
                       const std::vector<SMCProbeSetting> &settings,
                       std::vector<double> &out) {
  if (!smcCapable)
    return false;
  uint current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  visited[current] = true;
  std::vector<DCLinkedHedron *> seq;
  WalkEdges(current, seq);
  for (DCLinkedHedron *h : seq)
    h->CacheOldBonds(oldMol);

  const uint first0 = data.nLJTrialsFirst, nth0 = data.nLJTrialsNth;
  const uint minTrials0 = data.calc.SetLocalMinTrials(1);
  std::vector<GraphGrowth> ctx(1); // serial
  GraphGrowth &g = ctx[0];
  g.data = &data;
  g.start = nodes[current].starting;
  g.seq = &seq;
  g.molIndex = molIndex;
  out.clear();
  for (const SMCProbeSetting &st : settings) {
    g.k = st.k;
    g.buf.resize(st.k);
    data.EnsureLJCapacity(st.k);
    data.nLJTrialsFirst = data.nLJTrialsNth = st.k;
    SMCGrowth<GraphParticle, PRNG> smc(st.N, 2.0);
    auto t0 = std::chrono::steady_clock::now();
    out.push_back(smc.GrowNew(GraphParticle(ctx, newMol), data.prng));
    out.push_back(smc.WeighOld(GraphParticle(ctx, oldMol), data.prng));
    out.push_back(std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count());
  }
  data.nLJTrialsFirst = first0;
  data.nLJTrialsNth = nth0;
  data.calc.SetLocalMinTrials(minTrials0);
  return true;
}
#endif

void DCGraph::BuildEdges(TrialMol &oldMol, TrialMol &newMol, uint molIndex,
                         const uint cur) {
  uint current = cur;
  // Copy the edges of the node to fringe
  fringe = nodes[current].edges;
  // Advance along edges, building as we go
  while (!fringe.empty()) {
    uint pick = data.prng.randIntExc(fringe.size());
    DCComponent *comp = fringe[pick].component;
    // Call DCLinkedHedron and build all Atoms connected to selected edge
    comp->PrepareNew(newMol, molIndex);
    comp->BuildNew(newMol, molIndex);
    comp->PrepareOld(oldMol, molIndex);
    comp->BuildOld(oldMol, molIndex);

    // travel to new node, remove traversed edge
    // Current node is the edge that we picked
    current = fringe[pick].destination;
    // Remove the edge that we visited
    fringe[pick] = fringe.back();
    fringe.pop_back();
    // Visiting the node
    visited[current] = true;

    // add edges to unvisited nodes
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

void DCGraph::Regrowth(TrialMol &oldMol, TrialMol &newMol, uint molIndex) {
  // Randomly pick a node to keep it fixed and not grow it
  int current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  // Visiting the node
  visited[current] = true;
  // Copy the current node's focus coordinate
  uint seedInx = nodes[current].atomIndex;
  newMol.AddAtom(seedInx, oldMol.AtomPosition(seedInx));
  oldMol.ConfirmOldAtom(seedInx);
  // check if we want to grow all atoms from node's focus or not
  bool growAll = data.prng() < 1.0 / nodes.size();
  if (growAll) {
    DCComponent *comp = nodes[current].restarting;
    // Call DCFreeHedronSeed to build all Atoms connected to the node's focus
    comp->PrepareNew(newMol, molIndex);
    comp->BuildNew(newMol, molIndex);
    comp->PrepareOld(oldMol, molIndex);
    comp->BuildOld(oldMol, molIndex);
    // Build all edges
    BuildEdges(oldMol, newMol, molIndex, current);
  } else {
    // Copy the all atoms bonded to node's focus
    for (uint b = 0; b < nodes[current].partnerIndex.size(); b++) {
      uint partner = nodes[current].partnerIndex[b];
      newMol.AddAtom(partner, oldMol.AtomPosition(partner));
      oldMol.ConfirmOldAtom(partner);
    }
    // First we pick a edge that will be fixed and copy the coordinates
    // We continue the same until only one edge left from this node
    // If current is the terminal node, we don't enter the while loop
    // Then continue to build the rest of the molecule from current

    // Copy the edges of the node to currFringe
    currFringe = nodes[current].edges;
    while (currFringe.size() > 1) {
      // randomely pick one of the edges connected to current (fixNode)
      uint pickFixEdg = data.prng.randIntExc(currFringe.size());
      // Travel to picked edges and make it as new fixNode
      uint fixNode = currFringe[pickFixEdg].destination;
      visited[fixNode] = true;
      // Copy the all atoms bonded to fixNode's focus
      for (uint b = 0; b < nodes[fixNode].partnerIndex.size(); b++) {
        uint partner = nodes[fixNode].partnerIndex[b];
        newMol.AddAtom(partner, oldMol.AtomPosition(partner));
        oldMol.ConfirmOldAtom(partner);
      }
      // Copy the edges of the new node to fringe
      fringe = nodes[fixNode].edges;
      // remove the edge that we traveled from
      for (uint f = 0; f < fringe.size(); f++) {
        if (fringe[f].destination == current)
          fringe.erase(fringe.begin() + f);
      }
      // Continue along picked edges and copy the coordinates
      while (!fringe.empty()) {
        fixNode = fringe[0].destination;
        // Copy the all atoms bonded to fixNode's focus
        for (uint b = 0; b < nodes[fixNode].partnerIndex.size(); b++) {
          uint partner = nodes[fixNode].partnerIndex[b];
          newMol.AddAtom(partner, oldMol.AtomPosition(partner));
          oldMol.ConfirmOldAtom(partner);
        }
        // Travel to new fixNode, remove traversed edge
        fringe[0] = fringe.back();
        fringe.pop_back();
        visited[fixNode] = true;
        // Add edges to unvisited nodes
        for (uint i = 0; i < nodes[fixNode].edges.size(); ++i) {
          Edge &e = nodes[fixNode].edges[i];
          if (!visited[e.destination]) {
            fringe.push_back(e);
          }
        }
      }
      // Remove the fixed edge from currFring
      currFringe.erase(currFringe.begin() + pickFixEdg);
    }
    // Now Start building the rest of the molecule from current
    // Start with only one left edge
    // Advance along edges, building as we go
    while (!currFringe.empty()) {
      // Randomly pick one of the edges connected to node
      uint pick = data.prng.randIntExc(currFringe.size());
      DCComponent *comp = currFringe[pick].component;
      // Call DCLinkedHedron and build all Atoms connected to selected edge
      comp->PrepareNew(newMol, molIndex);
      comp->BuildNew(newMol, molIndex);
      comp->PrepareOld(oldMol, molIndex);
      comp->BuildOld(oldMol, molIndex);
      current = currFringe[pick].destination;
      // Remove the edge that we visited
      currFringe[pick] = currFringe.back();
      currFringe.pop_back();
      visited[current] = true;
      for (uint i = 0; i < nodes[current].edges.size(); ++i) {
        Edge &e = nodes[current].edges[i];
        if (!visited[e.destination]) {
          currFringe.push_back(e);
        }
      }
    }
  }
}

void DCGraph::BuildIDNew(TrialMol &newMol, uint molIndex) {
  idExchange->PrepareNew(newMol, molIndex);
  idExchange->BuildNew(newMol, molIndex);
}

void DCGraph::BuildIDOld(TrialMol &oldMol, uint molIndex) {
  idExchange->PrepareOld(oldMol, molIndex);
  idExchange->BuildOld(oldMol, molIndex);
}

void DCGraph::BuildOld(TrialMol &oldMol, uint molIndex) {
  // Randomly pick a node to call DCFreeHedron on it
  uint current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareOld(oldMol, molIndex);
  comp->BuildOld(oldMol, molIndex);
  // Advance along edges, building as we go
  // Copy the edges of the node to fringe
  fringe = nodes[current].edges;
  // Advance along edges, building as we go
  while (!fringe.empty()) {
    // Randomly pick one of the edges connected to node
    uint pick = data.prng.randIntExc(fringe.size());
    DCComponent *comp = fringe[pick].component;
    // Call DCLinkedHedron and build all Atoms connected to selected edge
    comp->PrepareOld(oldMol, molIndex);
    comp->BuildOld(oldMol, molIndex);

    // Travel to new node, remove traversed edge
    // Current node is the edge that we picked
    current = fringe[pick].destination;
    // Remove the edge that we visited
    fringe[pick] = fringe.back();
    fringe.pop_back();
    // Visiting the node
    visited[current] = true;

    // Add edges to unvisited nodes
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

void DCGraph::BuildNew(TrialMol &newMol, uint molIndex) {
  // Randomly pick a node to call DCFreeHedron on it
  uint current = data.prng.randIntExc(nodes.size());
  visited.assign(nodes.size(), false);
  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareNew(newMol, molIndex);
  comp->BuildNew(newMol, molIndex);
  // Advance along edges, building as we go
  // Copy the edges of the node to fringe
  fringe = nodes[current].edges;
  // Advance along edges, building as we go
  while (!fringe.empty()) {
    // Randomly pick one of the edges connected to node
    uint pick = data.prng.randIntExc(fringe.size());
    DCComponent *comp = fringe[pick].component;
    // Call DCLinkedHedron and build all Atoms connected to selected edge
    comp->PrepareNew(newMol, molIndex);
    comp->BuildNew(newMol, molIndex);

    // Travel to new node, remove traversed edge
    // Current node is the edge that we picked
    current = fringe[pick].destination;
    // Remove the edge that we visited
    fringe[pick] = fringe.back();
    fringe.pop_back();
    // Visiting the node
    visited[current] = true;

    // Add edges to unvisited nodes
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

void DCGraph::BuildGrowOld(TrialMol &oldMol, uint molIndex) {
  visited.assign(nodes.size(), false);
  // Use backbone atom to start the node
  int current = -1;
  for (int i = 0; i < (int)nodes.size(); i++) {
    if (nodes[i].atomIndex == oldMol.GetAtomBB(0)) {
      current = i;
      break;
    }
  }

  if (current == -1) {
    std::cout << "Error: In MEMC-3 move, atom "
              << oldMol.GetKind().atomNames[oldMol.GetAtomBB(0)] << " in "
              << oldMol.GetKind().name << " must be a node.\n";
    std::cout << "       This atom must be bounded to two or more atoms! \n";
    exit(1);
  }

  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareOld(oldMol, molIndex);
  comp->BuildOld(oldMol, molIndex);
  // Advance along edges, building as we go
  // Copy the edges of the node to fringe
  fringe = nodes[current].edges;
  // Advance along edges, building as we go
  while (!fringe.empty()) {
    // Randomly pick one of the edges connected to node
    uint pick = data.prng.randIntExc(fringe.size());
    DCComponent *comp = fringe[pick].component;
    // Call DCLinkedHedron and build all Atoms connected to selected edge
    comp->PrepareOld(oldMol, molIndex);
    comp->BuildOld(oldMol, molIndex);

    // Travel to new node, remove traversed edge
    // Current node is the edge that we picked
    current = fringe[pick].destination;
    // Remove the edge that we visited
    fringe[pick] = fringe.back();
    fringe.pop_back();
    // Visiting the node
    visited[current] = true;

    // Add edges to unvisited nodes
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

void DCGraph::BuildGrowNew(TrialMol &newMol, uint molIndex) {
  visited.assign(nodes.size(), false);
  // Use backbone atom to start the node
  int current = -1;
  for (int i = 0; i < (int)nodes.size(); i++) {
    if (nodes[i].atomIndex == newMol.GetAtomBB(0)) {
      current = i;
      break;
    }
  }

  if (current == -1) {
    std::cout << "Error: In MEMC-3 move, atom "
              << newMol.GetKind().atomNames[newMol.GetAtomBB(0)] << " in "
              << newMol.GetKind().name << " must be a node.\n";
    std::cout << "       This atom must be bounded to two or more atoms! \n";
    exit(1);
  }

  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareNew(newMol, molIndex);
  comp->BuildNew(newMol, molIndex);
  // Advance along edges, building as we go
  // Copy the edges of the node to fringe
  fringe = nodes[current].edges;
  // Advance along edges, building as we go
  while (!fringe.empty()) {
    // Randomly pick one of the edges connected to node
    uint pick = data.prng.randIntExc(fringe.size());
    DCComponent *comp = fringe[pick].component;
    // Call DCLinkedHedron and build all Atoms connected to selected edge
    comp->PrepareNew(newMol, molIndex);
    comp->BuildNew(newMol, molIndex);

    // Travel to new node, remove traversed edge
    // Current node is the edge that we picked
    current = fringe[pick].destination;
    // Remove the edge that we visited
    fringe[pick] = fringe.back();
    fringe.pop_back();
    // Visiting the node
    visited[current] = true;

    // Add edges to unvisited nodes
    for (uint i = 0; i < nodes[current].edges.size(); ++i) {
      Edge &e = nodes[current].edges[i];
      if (!visited[e.destination]) {
        fringe.push_back(e);
      }
    }
  }
}

void DCGraph::BuildGrowInCav(TrialMol &oldMol, TrialMol &newMol,
                             uint molIndex) {
  visited.assign(nodes.size(), false);
  // Get the seedIndex
  int sIndex;
  if (newMol.HasCav()) {
    sIndex = newMol.GetGrowingAtomIndex();
  } else if (oldMol.HasCav()) {
    sIndex = oldMol.GetGrowingAtomIndex();
  } else {
    std::cout << "Error: Calling BuildGrowInCav, but there is no cavity"
              << " defined for newMol and oldMol.\n";
    exit(EXIT_FAILURE);
  }

  // Use backbone atom to start the node
  int current = -1;
  for (int i = 0; i < (int)nodes.size(); i++) {
    if ((int)nodes[i].atomIndex == sIndex) {
      current = i;
      break;
    }
  }

  if (current == -1) {
    std::cout << "Error: In TargetedSwap or IntraTargetedSwap move, atom "
              << newMol.GetKind().atomNames[sIndex] << " in "
              << newMol.GetKind().name << " must be a node.\n";
    std::cout << "       This atom must be bounded to two or more atoms! \n";
    exit(1);
  }

  // Visiting the node
  visited[current] = true;
  DCComponent *comp = nodes[current].starting;
  // Call DCFreeHedron to build all Atoms connected to the node
  comp->PrepareNew(newMol, molIndex);
  comp->BuildNew(newMol, molIndex);
  comp->PrepareOld(oldMol, molIndex);
  comp->BuildOld(oldMol, molIndex);
  // Advance along edges, building as we go
  BuildEdges(oldMol, newMol, molIndex, current);
}

DCGraph::~DCGraph() {
  for (SMCWorker &w : smcWorkers) {
    delete w.graph;
    delete w.prng; // and its generator
  }
  delete idExchange;
  delete recoil;
  for (uint v = 0; v < nodes.size(); ++v) {
    Node &node = nodes[v];
    delete node.starting;
    delete node.restarting;
    for (uint e = 0; e < node.edges.size(); ++e) {
      delete node.edges[e].component;
    }
  }

  for (uint i = 0; i < shaftNodes.size(); i++) {
    delete shaftNodes[i];
  }
}

} // namespace cbmc
