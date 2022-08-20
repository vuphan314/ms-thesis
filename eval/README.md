# Evaluation

--------------------------------------------------------------------------------

## File `data.csv`
There are 173 lines with `-1.0` as the `weightedModelCount`.
They indicate that overflow occurs when a model count produced by Cachet is multiplied by `2^n`,
where `n` is the number of variables in a benchmark.
(As mentioned in our paper, Cachet cannot accept literal weights `0.5` and `1.5`,
so we use weights `0.25` and `0.75` then multiply the model count by `2^n`.)
We regard these 173 instances as completions when comparing Cachet to other model counters
(so as not to overly disadvantage Cachet).

--------------------------------------------------------------------------------

## Dir `data/`
SLURM `.out` files

--------------------------------------------------------------------------------

## Dir `analysis/`
Figures and tables
