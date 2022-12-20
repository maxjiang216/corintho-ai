# corintho-ai

An AI that plays [Corintho](http://www.di.fc.ul.pt/~jpn/gv/corintho.htm). Uses the Alpha Zero Monte Carlo Tree Search and Neural Network position evaluation method based on [this paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

# To-do

- [x] Implement Corintho's rules
- [x] Implement MCTS
- [x] Optimize legal move generator
- [x] Add neural network for evaluation
- [x] Implement concurrent gameplay and batch evaluation for neural network
- [x] Make neural network training pipeline
- [x] Optimize code using C++ and Cython
- [x] Make config parser for hyperparameters and learning rate schedule
- [x] Add training on samples from previous generations
- [ ] Add Leela-Zero-style nodes
- [ ] Add validation tests
- [ ] Check for memory leaks
- [ ] Add board symmetries in training samples
- [ ] Make testing such that only only neural network is queried each time
- [ ] Remove offset for searches during testing
- [ ] Validate correctness of Dirichlet distribution approximation
- [ ] Add the option for multiple searches per evaluation
- [ ] Add Boost thread pool for multithreading
- [ ] Profile code
- [ ] Add Boost memory pool for allocating nodes and edges
- [ ] Try tcmalloc for fasting dynamic allocation
- [ ] Train neural network
- [ ] Make front end
- [ ] Assess rating for best player
