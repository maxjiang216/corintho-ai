#include "selfplayer.h"

SelfPlayer::SelfPlayer(Trainer *trainer): testing{false}, logging{false},
players{TrainMC{trainer}, TrainMC{trainer}} {}

SelfPlayer::SelfPlayer(bool, Trainer *trainer): testing{false}, logging{true},
players{TrainMC{true, trainer}, TrainMC{true, trainer}} {}

SelfPlayer::SelfPlayer(bool logging, bool seed, Trainer *trainer): testing{true}, logging{logging}, seed{seed},
players{TrainMC{true, logging, trainer}, TrainMC{true, logging, trainer}} {}

// It is relatively costless to detect when a SelfPlayer will be called by Trainer for the first time
// Instead of having an extra if statement in do_iteration, we can split it into 2 functions
void SelfPlayer::do_first_iteration() {
    players[0].do_first_iteration();
}