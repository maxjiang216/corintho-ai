#include "selfplayer.h"

SelfPlayer::SelfPlayer(): testing{false}, logging{false} {}

SelfPlayer::SelfPlayer(bool logging): testing{false}, logging{logging} {}

SelfPlayer::SelfPlayer(bool logging, bool seed): testing{true}, logging{logging}, seed{seed} {}

// It is relatively costless to detect when a SelfPlayer will be called by Trainer for the first time
// Instead of having an extra if statement in do_iteration, we can split it into 2 functions
void SelfPlayer::do_first_iteration() {
    players[0].do_first_iteration();
}