#include "selfplayer.h"

SelfPlayer::SelfPlayer(): testing{false}, logging{false} {}

SelfPlayer::SelfPlayer(bool logging): testing{false}, logging{logging} {}

SelfPlayer::SelfPlayer(bool logging, bool seed): testing{true}, logging{logging}, seed{seed} {}

