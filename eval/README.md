# Evaluation (Linux)

--------------------------------------------------------------------------------

## Benchmarks

### Downloading archive to this dir (`eval`)
```bash
wget https://github.com/vuphan314/ms-thesis/releases/download/v0/benchmarks.zip
```

### Extracting downloaded archive into new dir `eval/benchmarks`
```bash
unzip benchmarks.zip
```

### Dir `eval/benchmarks/altogether`
1914 CNF formulas:
- 1091 from Bayesian inference
- 823 from other applications

--------------------------------------------------------------------------------

## Data

### Downloading archive to this dir (`eval`)
```bash
wget https://github.com/vuphan314/ms-thesis/releases/download/v0/data.zip
```

### Extracting downloaded archive into dir `eval/data`
```bash
unzip data.zip
```

### Dir `eval/data/altogether`
SLURM `.out` files

### File [`data.csv`](./data.csv)
There are 173 lines with `-1.0` as the `weightedModelCount`.
They indicate that overflow occurs when a model count produced by Cachet is multiplied by `2^n`,
where `n` is the number of variables in a benchmark.
(As mentioned in our paper, Cachet cannot accept literal weights `0.5` and `1.5`,
so we use weights `0.25` and `0.75` then multiply the model count by `2^n`.)
We regard these 173 instances as completions when comparing Cachet to other model counters
(so as not to overly disadvantage Cachet).

--------------------------------------------------------------------------------

## Figures and tables
Dir [`analysis`](./analysis/)
