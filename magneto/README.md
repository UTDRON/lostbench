# Magneto Experiments on LostBench

To Run Magneto Matcher on the LostBech Dataset

## Setup

Copy the following files into the original Magneto repository:

- `anes_benchmark.py` → `magneto/experiments/benchmark/`
- `llm_reranker.py` → `magneto/algorithms/magneto/magneto/` (replace)
- `utils.py` → `magneto/algorithms/magneto/magneto/utils/` (replace)

## Run

```bash
python anes_benchmark.py --mode [MODE] --dataset anes
```

## Modes
`header-value-default`
`header-value-repeat`
`header-value-verbose`

## Note

Experimental results in the paper use:

`--mode header-value-verbose`
