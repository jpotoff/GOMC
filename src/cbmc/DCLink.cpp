#include "DCLink.h" 
#include "TrialMol.h" 
#include "Forcefield.h" 
#include "XYZArray.h" 
#include "MoleculeKind.h" 
#include "MolSetup.h" 
#include "NumLib.h" 
 
namespace cbmc 
{ 
 
   DCLink::DCLink(DCData* data, const mol_setup::MolKind kind, 
		  uint atom, uint focus) 

     : data(data), atom(atom), focus(focus), angleFix(false), bondFix(false) 

   { 
      //will fail quietly if not a part of a valid linear molecule, 
      //but we checked that, right? 
      using namespace mol_setup; 
      std::vector<Bond> bonds = AtomBonds(kind, atom); 
      for (uint i = 0; i < bonds.size(); ++i) 
      { 
         if (bonds[i].a0 == focus || bonds[i].a1 == focus) 
	 { 
            eqBondLength = data->ff.bonds.Length(bonds[i].kind);  
	    bondKind = bonds[i].kind; 
	    bondFix = data->ff.bonds.bondFixed(bondKind);
            break; 
         } 
      } 
 
      std::vector<Angle> angles = AtomEndAngles(kind, atom); 
      for (uint i = 0; i < angles.size(); ++i) 
      { 
	 if (angles[i].a1 == focus) 
	 { 
	    angleKind = angles[i].kind;
	    
	    if (data->ff.angles->AngleFixed(angleKind))
	    {
	      angleFix = true;
	      thetaFix = data->ff.angles->Angle(angleKind);
	    }

            break; 
	 } 
      } 
      std::vector<Dihedral> dihs = AtomEndDihs(kind, atom); 
      for (uint i = 0; i < dihs.size(); ++i) 
      { 
         if (dihs[i].a1 == focus) 
	 { 
            dihKind = dihs[i].kind; 
            prev = dihs[i].a2; 
            prevprev = dihs[i].a3; 
            break; 
         } 
      } 
   } 
 
   void DCLink::PrepareNew(TrialMol& newMol, uint molIndex) 
   { 
      double* angles = data->angles; 
      double* angleEnergy = data->angleEnergy; 
      double* angleWeights = data->angleWeights; 
      double* nonbonded_1_3 =  data->nonbonded_1_3; 
      PRNG& prng = data->prng; 
      const Forcefield& ff = data->ff; 
      uint count = data->nAngleTrials; 
      std::fill_n(nonbonded_1_3, count, 0.0); 
      bendWeight = 0; 

      SetNewBond(newMol);
 
      for (uint trial = 0; trial < count; trial++) 
      { 
	 if (angleFix) 
	 { 
	    angles[trial] = thetaFix; 
	 } 
	 else 
	 {	    
	    angles[trial] = prng.rand(M_PI); 	    
	 } 
	 angleEnergy[trial] = ff.angles->Calc(angleKind, angles[trial]); 

	 double distSq = newMol.AngleDist(bond[1], bond[2], angles[trial]); 
	 nonbonded_1_3[trial] = data->calc.IntraEnergy_1_3(distSq, prev, atom, 
							   molIndex); 
 
	 if(isnan(nonbonded_1_3[trial])) 
	    nonbonded_1_3[trial] = num::BIGNUM; 
 
         angleWeights[trial] = exp((angleEnergy[trial] + nonbonded_1_3[trial]) 
				   * -ff.beta); 
 
         bendWeight += angleWeights[trial]; 
      } 
      uint winner = prng.PickWeighted(angleWeights, count, bendWeight); 
      theta = angles[winner]; 
      bendEnergy = angleEnergy[winner]; 
      oneThree = nonbonded_1_3[winner]; 
   } 
 
   void DCLink::PrepareOld(TrialMol& oldMol, uint molIndex) 
   { 
      PRNG& prng = data->prng; 
      const Forcefield& ff = data->ff; 
      uint count = data->nAngleTrials - 1; 
      bendWeight = 0; 
 
      //set bond distance for old molecule 
      SetOldBond(oldMol); 
 
      for (uint trial = 0; trial < count; trial++) 
      { 
	 double trialAngle; 
	 double trialEn; 
 
	 if(angleFix) 
	 { 
	   trialAngle = thetaFix; 
	 } 
	 else 
	 {	    
	   trialAngle = prng.rand(M_PI); 
	 } 
	 trialEn = ff.angles->Calc(angleKind, trialAngle);

 	 double distSq = oldMol.AngleDist(oldBond[1], oldBond[2], trialAngle);  
	 double tempEn = data->calc.IntraEnergy_1_3(distSq, prev, atom,
						    molIndex); 
	 if(isnan(tempEn)) 
	    tempEn = num::BIGNUM; 
 
         trialEn += tempEn; 

         double trialWeight = exp(-ff.beta * trialEn); 
         bendWeight += trialWeight; 
      } 
   } 
 
   void DCLink::IncorporateOld(TrialMol& oldMol, uint molIndex) 
   { 
      oldMol.OldThetaAndPhi(atom, focus, theta, phi); 
      const Forcefield& ff = data->ff; 

      bendEnergy = ff.angles->Calc(angleKind, theta); 
      double distSq = oldMol.GetDistSq(prev, atom); 
      oneThree = data->calc.IntraEnergy_1_3(distSq, prev, atom, molIndex); 
      bendWeight += exp(-ff.beta * (bendEnergy + oneThree));  
   } 
 
   void DCLink::AlignBasis(TrialMol& mol) 
   { 
      mol.SetBasis(focus, prev, prevprev); 
   } 
 
   void DCLink::SetOldBond(TrialMol& oldMol) 
   { 
     double BondDistSq1 = oldMol.GetDistSq(focus, atom); 
     double BondDistSq2 = oldMol.GetDistSq(prev, focus); 
     double BondDistSq3 = oldMol.GetDistSq(prevprev, prev); 
     oldBond[2] = sqrt(BondDistSq1);
     oldBond[1] = sqrt(BondDistSq2);
     oldBond[0] = sqrt(BondDistSq3);

     //bond length from focus to atom
     oldBondLength = oldBond[2];
     oldBondEnergy = data->ff.bonds.Calc(bondKind, oldBondLength);
     oldBondWeight = exp(-1 * data->ff.beta * oldBondEnergy);
   } 

   void DCLink::SetNewBond(TrialMol& newMol) 
   {
     if(bondFix)
     {
       newBondLength = eqBondLength;
       newBondEnergy = data->ff.bonds.Calc(bondKind, newBondLength);
       newBondWeight = exp(-1 * data->ff.beta * newBondEnergy);
     }
     else
     {
        double bond, bf;
        do
        {
	   do
	   {
	      bond = 0.2 * data->prng.rand() + 0.9;
	      bf = bond * bond * bond / 1.331;

	   }while(bf < data->prng.rand());

	   newBondLength = bond * eqBondLength;
	   newBondEnergy = data->ff.bonds.Calc(bondKind, newBondLength);
	   newBondWeight = exp(-1 * data->ff.beta * newBondEnergy);  
 
	}while(newBondWeight < data->prng.rand());
     }
     bond[2] = newBondLength;
     bond[1] = sqrt(newMol.GetDistSq(prev, focus));
     bond[0] = sqrt(newMol.GetDistSq(prevprev, prev));
   }
 
   void DCLink::BuildOld(TrialMol& oldMol, uint molIndex) 
   { 
      AlignBasis(oldMol); 
      IncorporateOld(oldMol, molIndex); 
      double* angles = data->angles; 
      double* angleEnergy = data->angleEnergy; 
      double* angleWeights = data->angleWeights; 
      double* ljWeights = data->ljWeights; 
      double* torsion = data->bonded; 
      double* nonbonded = data->nonbonded; 
      double* nonbonded_1_4 = data->nonbonded_1_4; 
      double* inter = data->inter; 
      double* real = data->real; 
      double* self = data->self; 
      double* correction = data->correction; 
      double* oneFour = data->oneFour; 
      uint nLJTrials = data->nLJTrialsNth; 
      XYZArray& positions = data->positions; 
      PRNG& prng = data->prng; 
 
      std::fill_n(inter, nLJTrials, 0.0); 
      std::fill_n(nonbonded, nLJTrials, 0.0); 
      std::fill_n(nonbonded_1_4, nLJTrials, 0.0); 
      std::fill_n(self, nLJTrials, 0.0); 
      std::fill_n(real, nLJTrials, 0.0); 
      std::fill_n(correction, nLJTrials, 0.0); 
      std::fill_n(ljWeights, nLJTrials, 0.0); 
      std::fill_n(angles, data->nDihTrials, 0.0); 
      std::fill_n(angleWeights, data->nDihTrials, 0.0); 
      std::fill_n(angleEnergy, data->nDihTrials, 0.0); 
      std::fill_n(torsion, nLJTrials, 0.0); 
      std::fill_n(oneFour, nLJTrials, 0.0); 
 
      UseOldDih(oldMol, molIndex, torsion[0], ljWeights[0]); 
      oneFour[0] = nonbonded_1_4[0]; 
      positions.Set(0, oldMol.AtomPosition(atom)); 
       
      for (uint trial = 1; trial < nLJTrials; ++trial) 
      { 
	ljWeights[trial] = GenerateDihedralsOld(oldMol, molIndex, angles, 
						angleEnergy, angleWeights); 
         uint winner = prng.PickWeighted(angleWeights, data->nDihTrials, 
					ljWeights[trial]); 
         torsion[trial] = angleEnergy[winner]; 
	 oneFour[trial] = nonbonded_1_4[winner]; 
         positions.Set(trial, oldMol.GetRectCoords(oldBondLength, theta, 
						   angles[winner])); 
      } 
 
      data->axes.WrapPBC(positions, oldMol.GetBox()); 
      data->calc.ParticleInter(inter, real, positions, atom, molIndex, 
                               oldMol.GetBox(), nLJTrials); 

#ifdef _OPENMP
#pragma omp parallel sections
#endif
{  
#ifdef _OPENMP     
#pragma omp section
#endif
      data->calc.ParticleNonbonded(nonbonded, oldMol, positions, atom, 
				   oldMol.GetBox(), nLJTrials); 
#ifdef _OPENMP 
#pragma omp section
#endif
      data->calcEwald->SwapSelf(self, molIndex, atom, oldMol.GetBox(), 
			       nLJTrials); 
#ifdef _OPENMP 
#pragma omp section
#endif
      data->calcEwald->SwapCorrection(correction, oldMol, positions, atom,  
				     oldMol.GetBox(), nLJTrials); 
}
 
      const MoleculeKind& thisKind = oldMol.GetKind(); 
      double tempEn = 0.0; 
      for (uint i = 0; i < thisKind.NumAtoms(); i++) 
      { 
	 if (oldMol.AtomExists(i) && i != atom) 
	 { 
	    double distSq = oldMol.GetDistSq(i, atom); 
	    tempEn += data->calcEwald->CorrectionOldMol(oldMol, distSq, 
							     i, atom); 
	 } 
      } 
      correction[0] = tempEn; 
 

 
      double dihLJWeight = 0; 
      for (uint trial = 0; trial < nLJTrials; ++trial) 
      { 
         ljWeights[trial] *= exp(-data->ff.beta * 
				 (inter[trial] + real[trial] + 
				  nonbonded[trial] + self[trial] + 
				  correction[trial])); 
         dihLJWeight += ljWeights[trial]; 
      } 
      oldMol.MultWeight(dihLJWeight * bendWeight * oldBondWeight); 
      oldMol.ConfirmOldAtom(atom); 
      oldMol.AddEnergy(Energy(torsion[0] + bendEnergy + oldBondEnergy, 
			      nonbonded[0] + oneThree + oneFour[0], 
			      inter[0], real[0], 0.0, self[0], correction[0])); 
   } 
 
   void DCLink::BuildNew(TrialMol& newMol, uint molIndex) 
   { 
      AlignBasis(newMol); 
      double* angles = data->angles; 
      double* angleEnergy = data->angleEnergy; 
      double* angleWeights = data->angleWeights; 
      double* ljWeights = data->ljWeights; 
      double* torsion = data->bonded; 
      double* nonbonded = data->nonbonded; 
      double* nonbonded_1_4 = data->nonbonded_1_4; 
      double* inter = data->inter; 
      double* real = data->real; 
      double* self = data->self; 
      double* correction = data->correction; 
      double* oneFour = data->oneFour; 
      uint nLJTrials = data->nLJTrialsNth; 
      XYZArray& positions = data->positions; 
      PRNG& prng = data->prng; 
 
      std::fill_n(inter, nLJTrials, 0.0); 
      std::fill_n(nonbonded, nLJTrials, 0.0); 
      std::fill_n(nonbonded_1_4, nLJTrials, 0.0); 
      std::fill_n(self, nLJTrials, 0.0); 
      std::fill_n(real, nLJTrials, 0.0); 
      std::fill_n(correction, nLJTrials, 0.0); 
      std::fill_n(ljWeights, nLJTrials, 0.0); 
      std::fill_n(angles, data->nDihTrials, 0.0); 
      std::fill_n(angleWeights, data->nDihTrials, 0.0); 
      std::fill_n(angleEnergy, data->nDihTrials, 0.0); 
      std::fill_n(torsion, nLJTrials, 0.0); 
      std::fill_n(oneFour, nLJTrials, 0.0); 
 
      for (uint trial = 0; trial < nLJTrials; ++trial) 
      { 
	ljWeights[trial] = GenerateDihedralsNew(newMol, molIndex, angles, 
						angleEnergy, angleWeights); 
         uint winner = prng.PickWeighted(angleWeights, data->nDihTrials, 
					 ljWeights[trial]); 
	 oneFour[trial] = nonbonded_1_4[winner]; 
         torsion[trial] = angleEnergy[winner]; 
         positions.Set(trial, newMol.GetRectCoords(newBondLength, theta,  
						   angles[winner])); 
      } 
 
      data->axes.WrapPBC(positions, newMol.GetBox()); 
      data->calc.ParticleInter(inter, real, positions, atom, molIndex, 
                               newMol.GetBox(), nLJTrials); 

#ifdef _OPENMP
#pragma omp parallel sections
#endif
{  
#ifdef _OPENMP     
#pragma omp section
#endif
      data->calc.ParticleNonbonded(nonbonded, newMol, positions, atom, 
				   newMol.GetBox(), nLJTrials); 
#ifdef _OPENMP     
#pragma omp section
#endif
      data->calcEwald->SwapSelf(self, molIndex, atom, newMol.GetBox(), 
			       nLJTrials); 
#ifdef _OPENMP     
#pragma omp section
#endif
      data->calcEwald->SwapCorrection(correction, newMol, positions, atom,  
				     newMol.GetBox(), nLJTrials); 
}

  
 
      double dihLJWeight = 0; 
      double beta = data->ff.beta; 
      for (uint trial = 0; trial < nLJTrials; ++trial) 
      { 
         ljWeights[trial] *= exp(-data->ff.beta * 
				 (inter[trial] + real[trial] + 
				  nonbonded[trial] + self[trial] + 
				  correction[trial])); 
 
         dihLJWeight += ljWeights[trial]; 
      } 
 
      uint winner = prng.PickWeighted(ljWeights, nLJTrials, dihLJWeight); 
      newMol.MultWeight(dihLJWeight * bendWeight * newBondWeight); 
      newMol.AddAtom(atom, positions[winner]); 
      newMol.AddEnergy(Energy(torsion[winner] + bendEnergy + newBondEnergy,
			      nonbonded[winner] + oneThree + oneFour[winner],
			      inter[winner], real[winner], 0.0, self[winner], 
			      correction[winner])); 
   } 
 
   double DCLink::GenerateDihedralsNew(TrialMol& newMol, uint molIndex,  
				       double* angles, double* angleEnergy, 
				       double* angleWeights) 
   { 
      double* nonbonded_1_4 = data->nonbonded_1_4;      
      double stepWeight = 0.0; 
      PRNG& prng = data->prng; 
      const Forcefield& ff = data->ff; 
      std::fill_n(nonbonded_1_4, data->nDihTrials, 0.0); 
       
      double theta1 = newMol.GetTheta(prevprev, prev, focus); 
 
      for (uint trial = 0, count = data->nDihTrials; trial < count; ++trial) 
      { 
         angles[trial] = prng.rand(2 * M_PI); 
         angleEnergy[trial] = ff.dihedrals.Calc(dihKind, angles[trial]); 
	 double distSq = newMol.DihedDist(bond[0], bond[1], bond[2], theta1, 
					  theta, angles[trial]); 
	 nonbonded_1_4[trial] = data->calc.IntraEnergy_1_4(distSq, prevprev, 
							   atom, molIndex); 
	 if(isnan(nonbonded_1_4[trial])) 
	    nonbonded_1_4[trial] = num::BIGNUM; 
 
	 angleWeights[trial] = exp(-ff.beta * (angleEnergy[trial] + 
					       nonbonded_1_4[trial])); 
         stepWeight += angleWeights[trial]; 
      } 
      return stepWeight; 
   } 
 
   double DCLink::GenerateDihedralsOld(TrialMol& oldMol, uint molIndex,  
				       double* angles, double* angleEnergy, 
				       double* angleWeights) 
   { 
      double* nonbonded_1_4 = data->nonbonded_1_4;      
      double stepWeight = 0.0; 
      PRNG& prng = data->prng; 
      const Forcefield& ff = data->ff; 
      std::fill_n(nonbonded_1_4, data->nDihTrials, 0.0); 
       
      double theta1 = oldMol.GetTheta(prevprev, prev, focus); 
 
      for (uint trial = 0, count = data->nDihTrials; trial < count; ++trial) 
      { 
         angles[trial] = prng.rand(2 * M_PI); 
         angleEnergy[trial] = ff.dihedrals.Calc(dihKind, angles[trial]); 
	 double distSq = oldMol.DihedDist(oldBond[0], oldBond[1], oldBond[2], 
					  theta1, theta, angles[trial]); 
  
	 nonbonded_1_4[trial] = data->calc.IntraEnergy_1_4(distSq, prevprev, 
							   atom, molIndex); 
 
	 if(isnan(nonbonded_1_4[trial])) 
	    nonbonded_1_4[trial] = num::BIGNUM; 
 
	 angleWeights[trial] = exp(-ff.beta * (angleEnergy[trial] + 
					       nonbonded_1_4[trial])); 
         stepWeight += angleWeights[trial]; 
      } 
      return stepWeight; 
   } 
 
   void DCLink::UseOldDih(TrialMol& oldMol, uint molIndex, double& energy, 
			  double& weight) 
   { 
      double* nonbonded_1_4 = data->nonbonded_1_4; 
      PRNG& prng = data->prng; 
      const Forcefield& ff = data->ff; 
      double beta = data->ff.beta; 
      std::fill_n(nonbonded_1_4, data->nDihTrials, 0.0); 
 
      double theta0 = oldMol.GetTheta(prevprev, prev, focus); 
 
      energy = ff.dihedrals.Calc(dihKind, phi); 
      double distSq = oldMol.GetDistSq(prevprev, atom); 
 
      nonbonded_1_4[0] = data->calc.IntraEnergy_1_4(distSq, prevprev, atom, 
						    molIndex); 
      weight = exp(-beta * (energy + nonbonded_1_4[0])); 
 
      for (uint trial = data->nDihTrials - 1; trial-- > 0;) 
      { 
         double trialPhi = prng.rand(2 * M_PI); 
	 double distSq = oldMol.DihedDist(oldBond[0], oldBond[1], oldBond[2], 
					  theta0, theta, trialPhi); 
 
	 double tempEn = data->calc.IntraEnergy_1_4(distSq, prevprev, atom,
						    molIndex); 
	 if(isnan(tempEn)) 
	    tempEn = num::BIGNUM; 
	  
         double trialEnergy = ff.dihedrals.Calc(dihKind, trialPhi) + tempEn; 
         weight += exp(-beta * trialEnergy); 
      } 
   } 
 
}             
