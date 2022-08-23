# corintho-ai

An AI that plays [Corintho](http://www.di.fc.ul.pt/~jpn/gv/corintho.htm). Uses the Alpha Zero Monte Carlo Tree Search and Neural Network position evaluation method based on [this paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

# To-do

- [x] Implement Corintho's rules
- [x] Implement MCTS
- [x] Optimize legal move generator
- [x] Add neural network for evaluation
- [x] Implement concurrent gameplay and batch evaluation for neural network
- [x] Make neural network training pipeline
- [ ] Train neural network

# Possible bonuses

- [ ] Test hyperparameters (learning rate)
- [ ] Make player share MCST and game state during training
- [ ] Test a convolutional network
- [ ] Test larger models
