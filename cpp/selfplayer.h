#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include "trainmc.h"

class SelfPlayer {

    bool testing, logging;
    TrainMC players[2];

  public:
    
    // Training mode
    SelfPlayer();
    // Testing mode
    SelfPlayer(int);
    ~SelfPlayer() = default;

    play();

};

#endif