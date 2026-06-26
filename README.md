# rsomics-chi-square

Chi-square test of independence on a contingency table — a value-exact Rust
reimplementation of `scipy.stats.chi2_contingency`, with the G-test
(log-likelihood ratio) variant and a 2×2 Fisher exact test
(`scipy.stats.fisher_exact`).

## Usage

```text
rsomics-chi-square <TABLE> [--no-correction] [--gtest] [--exact]
```

`TABLE` is a TSV grid of non-negative integer counts: one table row per line,
columns tab-separated. `-` or an omitted argument reads stdin.

| flag | effect |
|------|--------|
| *(default)* | Pearson chi-square with Yates' continuity correction (only acts on df==1 tables) |
| `--no-correction` | disable Yates' correction |
| `--gtest` | use the G-test (log-likelihood ratio) statistic instead of Pearson |
| `--exact` | Fisher's exact test (table must be 2×2) — two-sided p-value |

Output is `chi2<TAB>df<TAB>p`, or under `--exact` `oddsratio<TAB>p`.

```console
$ printf '10\t20\n30\t40\n' | rsomics-chi-square --no-correction
0.7936507936507936	1	0.37299848361348686

$ printf '8\t2\n1\t5\n' | rsomics-chi-square --exact
20	0.034965034965034975
```

`-t/--threads`, `-q/--quiet`, `--json` are provided by the shared CLI scaffold.

## Method

- **Expected frequencies**: `E_ij = (row_sum_i · col_sum_j) / N`.
- **Pearson chi-square**: `Σ (O − E)² / E`.
- **G-test** (`--gtest`): `2 · Σ O · ln(O / E)` (the term is 0 when `O == 0`).
- **Yates' correction**: applies only when `df == 1`, adjusting each observed
  cell toward its expectation by `min(0.5, |E − O|)` — SciPy's exact rule, which
  is gentler than the textbook "subtract 0.5 from `|O − E|`" when a cell already
  lies within 0.5 of `E`.
- **Degrees of freedom**: `(r − 1)(c − 1)`.
- **p-value**: `chi2.sf(statistic, df)`, computed through a Cephes `igam`/`igamc`
  port so the upper-tail incomplete gamma matches SciPy's special-function path.
- **Fisher exact 2×2** (`--exact`): the two-sided p-value as the sum of
  hypergeometric probabilities no greater than the observed table's probability,
  reproducing SciPy's binary-search of the opposite tail; the odds ratio is the
  sample ratio `ad/bc`.

### Relationship to other rsomics Fisher tools

`rsomics-bed-fisher` and `rsomics-vcf-contrast` also compute a Fisher exact
test, but in domain-specific contexts (BED interval-overlap enrichment, and
per-variant VCF case/control allelic contrast respectively). This crate's
`--exact` is a generic 2×2 contingency-table Fisher test matching SciPy's
two-sided convention; chi-square independence on an arbitrary `r × c` table is
the primary operation here.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.chi2_contingency` and `scipy.stats.fisher_exact` based on:

- Pearson's chi-square test of independence and the G-test / log-likelihood
  ratio (power-divergence family; Cressie & Read 1984).
- Fisher, R.A. (1922). *On the interpretation of χ² from contingency tables*.
  J. Royal Statistical Society 85(1):87–94.
- The public SciPy API behaviour, observed black-box and verified value-exact.
- A direct port of the Cephes incomplete-gamma functions (`igam`/`igamc`) for
  the chi-squared survival function, matching SciPy's `chdtrc`.

Test fixtures are independently generated and their reference statistics
computed once with SciPy 1.17.1; the compat test asserts agreement without
invoking SciPy.

License: MIT OR Apache-2.0.
Upstream credit: [SciPy](https://scipy.org) (BSD-3-Clause); Cephes special
functions by Stephen L. Moshier.
