![Corintho board game](/images/corintho.png)

# corintho-ai

An AI that plays [Corintho](http://www.di.fc.ul.pt/~jpn/gv/corintho.htm). Uses the AlphaZero Monte Carlo tree search and neural network evaluation method based on [this paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). We currently assess the model to play at a 1500 Elo rating.

# Technologies Used
 - C++ (Monte Carlo search tree & game logic)
 - Keras/Tensorflow (neural network architecture & training)
 - Python 3 (training pipeline)
 - Cython (facilitate C++ & Python interaction)
 - OpenMP (multithreading for C++ code)

Trained on Google Cloud Platform.
