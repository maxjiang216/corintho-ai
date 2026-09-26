# Research: what other AlphaZero-style projects use (2026-09-25)

Asked by the developer: network architectures and learning-rate schedules in
comparable projects. Sources at the end.

| project | network | optimizer / learning rate | data used for training | gating |
|---|---|---|---|---|
| AlphaZero (chess, shogi, Go) | ResNet, 256 filters, 20 blocks (19/39 in AGZ) | SGD, momentum 0.9, lr 0.2 -> 0.02 -> 0.002 -> 0.0002 at fixed step counts; batch 4096; weight decay 1e-4 | continuous training on recent self-play games | none (AlphaGo Zero had a 55% gate; AlphaZero dropped it) |
| KataGo (Go) | ResNet with global pooling; grown during the run: 6x96 -> 10x128 -> 15x192 -> 20x256 (later 28x512) | SGD, momentum 0.9, batch 256, fixed per-sample lr 6e-5 (2e-5 for the first 5M samples, 6e-6 at the end); L2 3e-5; weight averaging (EMA of snapshots, decay 0.75) | uniform samples from a moving window of the most recent data, 250k growing to ~22M samples | none |
| Leela Chess Zero | ResNet with squeeze-excitation (global pooling), 10x128 to 24x320; later transformers | lr starts high and is dropped a few times to near zero over a run | window of recent games (e.g. games 15M-67M for one run) | none |
| AlphaZero.jl, Connect Four | ResNet 5 blocks x 128 filters, 1.6M params | Adam lr 2e-3, L2 1e-4, batch 1024 | buffer 400k growing to 1M samples | arena check |
| alpha-zero-general, Othello | 4 conv + 2 FC (512 channels), dropout 0.3 | Adam lr 1e-3, batch 64, 10 epochs | recent iterations' examples | 60% arena gate |
| Jones, "Scaling Scaling Laws with Board Games" (Hex) | **fully-connected residual nets**; best for 9x9: 2 layers x 512 (~500k params) | lr 1e-3, batch 32k | buffer 2M samples | none |

Findings relevant here:

1. **Nobody anneals the learning rate on failed promotions.** Schedules
   follow training progress (steps or samples): a constant rate with a few
   drops (AlphaZero, Lc0, KataGo), or a constant Adam rate (AlphaZero.jl,
   alpha-zero-general, Hex). The old pipeline halved the rate on every
   rejection, from 1e-3 (gens 1-50) to 6.25e-5 (gen ~70) and 5e-6 (gen 90+).
2. **Training data is a sliding window over recent games**, sampled
   uniformly, not only the latest generation. The old `main.pyx` meant to
   use 2 older generations but discarded them (worklog 2026-09-20, entry
   29); the new pipeline uses `--old-gens 0`.
3. **Gating is mostly gone.** AlphaZero, KataGo and Lc0 always use the
   latest network. The gate plus annealing-on-rejection is what lets the
   rate spiral down.
4. **Weight averaging** (KataGo: EMA of snapshots) smooths the noisy
   self-play updates.
5. **Architecture:** residual nets with normalization everywhere, plus
   global pooling (KataGo, Lc0's squeeze-excitation) in the larger boards.
   On a small board, fully-connected residual nets are a reasonable choice
   (Hex scaling study; its best 9x9 net was 2 x 512, ~500k params, near our
   residual 256x4 at 574k). KataGo grew the network during the run.
6. **Auxiliary targets** help (KataGo: soft policy, opponent's reply,
   short-term value; 1.3x faster learning from auxiliary policy targets,
   1.6x from global pooling).
7. **Iterations over epochs** in small games (Wang et al.): many self-play
   iterations with few training epochs each, and too much training per
   iteration can hurt.
8. **Network size vs search** (Hex): 10x more training compute can replace
   ~15x test-time (search) compute. A larger network that costs more per
   evaluation can pay for itself with fewer searches.

Sources:
- AlphaZero hyperparameters (via the AlphaZero pseudocode and papers):
  https://www.bgonline.org/forums/webbbs_config.pl?noframes%3Bread=208877 ,
  https://medium.com/@umerhasan17/a-summary-of-the-general-reinforcement-learning-game-playing-algorithm-alphazero-755f1de1ce38
- KataGo paper: https://arxiv.org/abs/1902.10565 ; methods:
  https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- Lc0: https://lczero.org/blog/2018/10/lc0-training/ ,
  https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/
- AlphaZero.jl Connect Four:
  https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/
- alpha-zero-general:
  https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py
- Jones, Scaling Scaling Laws with Board Games: https://arxiv.org/abs/2104.03113
- Wang et al., hyper-parameters for small games: https://arxiv.org/abs/2003.05988 ,
  https://arxiv.org/abs/1903.08129
