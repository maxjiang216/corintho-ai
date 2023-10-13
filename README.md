<img src="/assets/images/logo.png" width="271" height="259">

# corintho-ai

An AI that plays [Corintho](http://www.di.fc.ul.pt/~jpn/gv/corintho.htm). Uses the AlphaZero Monte Carlo tree search and neural network evaluation method based on [this paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). We assess the AI to play at a 3780 Elo rating.

# Technologies Used
 - C++ (Monte Carlo search tree & game logic)
 - Keras/Tensorflow (neural network architecture & training)
 - Python 3 (training pipeline)
 - Cython (C++ & Python interaction)
 - OpenMP (multithreading in C++)

Trained on Google Cloud Platform for about 80 hours.

# Acknowledgements

This project makes use of the following third-party libraries:

- [Cython](https://github.com/cython/cython) - Licensed under the Apache License 2.0
- [keras](https://github.com/keras-team/keras) - Licensed under the MIT License
- [Microsoft GSL](https://github.com/microsoft/GSL) - Licensed under the MIT License
- [numpy](https://github.com/numpy/numpy) - Licensed under the BSD License
- [setuptools](https://github.com/pypa/setuptools) - Licensed under the MIT License
- [toml](https://github.com/uiri/toml) - Licensed under the MIT License

Please refer to each library's linked repository for the full license text.
