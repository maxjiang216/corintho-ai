# corintho-ai

An AI that plays [Corintho](http://www.di.fc.ul.pt/~jpn/gv/corintho.htm). Uses the Alpha Zero Monte Carlo Tree Search and Neural Network position evaluation method.

# To-do

- [x] Implement Corintho's rules
- [x] Implement MCTS
- [x] Optimize legal move generator
- [x] Add neural network for evaluation
- [ ] Add embedded layer to neural network for legal move masking
- [ ] Create Player class with neural network for evaulation and move guider
- [ ] Implement concurrent gameplay and batch evaluation for neural network
- [ ] Make neural network training pipeline
- [ ] Test hyperparameters (learning rate)
- [ ] Train neural network

# Possible bonuses

- [ ] Make player share MCST and game state during training
- [ ] Test a convolutional network
- [ ] Test larger models
