/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.50
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in License.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#ifndef RNGIDENTIFIERS_H
#define RNGIDENTIFIERS_H

#include <cstdint>


struct RNGIdentifier
    {
    //MultiParticleBrownian
    static const uint8_t MPBrownian_PickBox=0;
    static const uint8_t MPBrownian_moveType=1;
    static const uint8_t MPBrownian_moveTypeNEMTMC=2;
    static const uint8_t MPBrownian_AcceptMove=3;
    //System
    static const uint8_t System_PickMove=4;

    //Translate
    static const uint8_t Translate_Prep=5;
    static const uint8_t Translate_AcceptMove=6;

    };

#endif /*RNGIDENTIFIERS_H*/
