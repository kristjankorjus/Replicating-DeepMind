/* *****************************************************************************
 * A.L.E (Atari 2600 Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf
 * Released under GNU General Public License www.gnu.org/licenses/gpl-3.0.txt
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  Constants.cpp
 *
 *  Defines a set of constants used by various parts of the player agent code
 *
 **************************************************************************** */

#include "Constants.h"

std::string action_to_string(Action a) {
    static string tmp_action_to_string[] = {
        "PLAYER_A_NOOP"           //  0
        ,"PLAYER_A_FIRE"          //  1
        ,"PLAYER_A_UP"            //  2
        ,"PLAYER_A_RIGHT"         //  3
        ,"PLAYER_A_LEFT"          //  4
        ,"PLAYER_A_DOWN"          //  5
        ,"PLAYER_A_UPRIGHT"       //  6
        ,"PLAYER_A_UPLEFT"        //  7
        ,"PLAYER_A_DOWNRIGHT"     //  8
        ,"PLAYER_A_DOWNLEFT"      //  9
        ,"PLAYER_A_UPFIRE"        // 10
        ,"PLAYER_A_RIGHTFIRE"     // 11
        ,"PLAYER_A_LEFTFIRE"      // 12
        ,"PLAYER_A_DOWNFIRE"      // 13
        ,"PLAYER_A_UPRIGHTFIRE"   // 14
        ,"PLAYER_A_UPLEFTFIRE"    // 15
        ,"PLAYER_A_DOWNRIGHTFIRE" // 16
        ,"PLAYER_A_DOWNLEFTFIRE"  // 17
        ,"PLAYER_B_NOOP"          // 18
        ,"PLAYER_B_FIRE"          // 19
        ,"PLAYER_B_UP"            // 20
        ,"PLAYER_B_RIGHT"         // 21
        ,"PLAYER_B_LEFT"          // 22
        ,"PLAYER_B_DOWN"          // 23
        ,"PLAYER_B_UPRIGHT"       // 24
        ,"PLAYER_B_UPLEFT"        // 25
        ,"PLAYER_B_DOWNRIGHT"     // 26
        ,"PLAYER_B_DOWNLEFT"      // 27
        ,"PLAYER_B_UPFIRE"        // 28
        ,"PLAYER_B_RIGHTFIRE"     // 29
        ,"PLAYER_B_LEFTFIRE"      // 30
        ,"PLAYER_B_DOWNFIRE"      // 31
        ,"PLAYER_B_UPRIGHTFIRE"   // 32
        ,"PLAYER_B_UPLEFTFIRE"    // 33
        ,"PLAYER_B_DOWNRIGHTFIRE" // 34
        ,"PLAYER_B_DOWNLEFTFIRE"  // 35
        ,"__invalid__" // 36
        ,"__invalid__" // 37
        ,"__invalid__" // 38
        ,"__invalid__" // 39
        ,"RESET"       // 40
        ,"UNDEFINED"   // 41
        ,"RANDOM"      // 42
    };
    assert (a >= 0 && a <= 42);
    return tmp_action_to_string[a];
}
