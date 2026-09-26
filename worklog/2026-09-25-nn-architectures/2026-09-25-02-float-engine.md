# 02 — Float in selection and backup

2026-09-25. Commit `e436d1e`. Data: end of `data/gpu-limits.txt`.

The previous worklog listed "float instead of double in selection" as a
candidate needing a statistical check, because it is not bit-identical. The
developer judged it obviously safe; measured anyway.

There were no `double` variables left; the doubles were implicit, in
`trainmc.cpp`:
- `1.0` / `-1.0` literals in the selection scores (`-1.0 * eval / visits +
  weighted / (visits + 1.0)`), the backup (`cur_eval - 1.0`,
  `cur_eval *= -1.0`) and a few per-expansion divisions;
- unqualified `sqrt(float)`, which called the C `double sqrt`.

All became float (`1.0F`, `std::sqrt`). The remaining literals are
initializations or arguments to float parameters, converted at compile time.

Three interleaved pairs, real self-play (TensorRT, compact gen_93, 4000
games, seed = pair): engine -0.2%, -1.9%, -2.3% (~-1.5%). Wall unchanged:
self-play is GPU-bound (entry 01), so the engine waits a little longer.
Games: turns within 0.03%, first-player score unchanged. Unit tests 45/45.
