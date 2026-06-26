//! Fisher's exact test for a 2×2 contingency table, matching
//! `scipy.stats.fisher_exact` with the two-sided alternative.
//!
//! For a table [[a, b], [c, d]] SciPy parametrises the hypergeometric
//! distribution as M = a+b+c+d, n = a+b (first row total), N = a+c (first
//! column total), and the cell count a is the random variable. The two-sided
//! p-value is the sum of hypergeometric probabilities no greater than the
//! observed table's probability, computed via SciPy's binary-search of the
//! opposite tail rather than an explicit enumeration — reproduced here exactly.
//!
//! The odds ratio is the sample (conditional) ratio ad/bc, with the same
//! ±0 / ±∞ edge handling SciPy uses.

use serde::Serialize;

/// Result of a Fisher exact test on a 2×2 table.
#[derive(Debug, Clone, Serialize)]
pub struct FisherResult {
    /// Sample odds ratio ad/bc (∞ when the denominator is zero, NaN on a zero margin).
    pub odds_ratio: f64,
    /// Two-sided exact p-value.
    pub p: f64,
}

/// Hypergeometric log-pmf: log P(X = k) for X ~ Hypergeom(M, n, N).
///
/// k successes drawn, n total successes in the population of size M, N draws.
fn log_hypergeom_pmf(k: i64, m: i64, n: i64, n_draws: i64) -> f64 {
    let lo = (n_draws - (m - n)).max(0);
    let hi = n.min(n_draws);
    if k < lo || k > hi {
        return f64::NEG_INFINITY;
    }
    log_choose(n, k) + log_choose(m - n, n_draws - k) - log_choose(m, n_draws)
}

fn log_choose(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return f64::NEG_INFINITY;
    }
    libm::lgamma((n + 1) as f64) - libm::lgamma((k + 1) as f64) - libm::lgamma((n - k + 1) as f64)
}

fn hypergeom_pmf(k: i64, m: i64, n: i64, n_draws: i64) -> f64 {
    log_hypergeom_pmf(k, m, n, n_draws).exp()
}

/// Lower CDF P(X ≤ k).
fn hypergeom_cdf(k: i64, m: i64, n: i64, n_draws: i64) -> f64 {
    let lo = (n_draws - (m - n)).max(0);
    let hi = n.min(n_draws);
    let mut sum = 0.0;
    let mut i = lo;
    while i <= k.min(hi) {
        sum += hypergeom_pmf(i, m, n, n_draws);
        i += 1;
    }
    sum.min(1.0)
}

/// Survival function P(X > k).
fn hypergeom_sf(k: i64, m: i64, n: i64, n_draws: i64) -> f64 {
    let lo = (n_draws - (m - n)).max(0);
    let hi = n.min(n_draws);
    let mut sum = 0.0;
    let mut i = (k + 1).max(lo);
    while i <= hi {
        sum += hypergeom_pmf(i, m, n, n_draws);
        i += 1;
    }
    sum.min(1.0)
}

/// SciPy's implicit binary search: find i in [lo, hi] with a(i) ≤ d < a(i+1),
/// where `a` is monotone over [lo, hi].
fn binary_search<F: Fn(i64) -> f64>(a: F, d: f64, mut lo: i64, mut hi: i64) -> i64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let midval = a(mid);
        if midval < d {
            lo = mid + 1;
        } else if midval > d {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    if a(lo) <= d { lo } else { lo - 1 }
}

/// Two-sided Fisher exact test on a 2×2 table [[a, b], [c, d]].
#[must_use]
pub fn fisher_exact_2x2(a: i64, b: i64, c: i64, d: i64) -> FisherResult {
    // A zero in any row or column total: p = 1, odds ratio undefined.
    if (a + b) == 0 || (c + d) == 0 || (a + c) == 0 || (b + d) == 0 {
        return FisherResult {
            odds_ratio: f64::NAN,
            p: 1.0,
        };
    }

    let odds_ratio = if c > 0 && b > 0 {
        (a as f64 * d as f64) / (c as f64 * b as f64)
    } else {
        f64::INFINITY
    };

    let n1 = a + b;
    let n2 = c + d;
    let n = a + c;
    let m = n1 + n2;

    let pmf = |x: i64| hypergeom_pmf(x, m, n1, n);

    let mode = ((n + 1) * (n1 + 1)) / (n1 + n2 + 2);
    let pexact = pmf(a);
    let pmode = pmf(mode);

    let epsilon = 1e-14;
    let gamma = 1.0 + epsilon;

    let p = if (pexact - pmode).abs() / pexact.max(pmode) <= epsilon {
        1.0
    } else if a < mode {
        let plower = hypergeom_cdf(a, m, n1, n);
        if pmf(n) > pexact * gamma {
            plower
        } else {
            let guess = binary_search(|x| -pmf(x), -pexact * gamma, mode, n);
            plower + hypergeom_sf(guess, m, n1, n)
        }
    } else {
        let pupper = hypergeom_sf(a - 1, m, n1, n);
        if pmf(0) > pexact * gamma {
            pupper
        } else {
            let guess = binary_search(pmf, pexact * gamma, 0, mode);
            pupper + hypergeom_cdf(guess, m, n1, n)
        }
    };

    FisherResult {
        odds_ratio,
        p: p.min(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::fisher_exact_2x2;

    fn close(got: f64, want: f64, rel: f64) {
        let d = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(d <= rel, "got {got:e} want {want:e} rel {d:e} > {rel:e}");
    }

    #[test]
    fn upper_tail_case() {
        // [[8,2],[1,5]] -> scipy or=20.0, p=0.034965034965034975
        let r = fisher_exact_2x2(8, 2, 1, 5);
        close(r.odds_ratio, 20.0, 1e-12);
        close(r.p, 0.034_965_034_965_034_975, 1e-12);
    }

    #[test]
    fn lower_tail_case() {
        // [[1,9],[11,3]] -> scipy or=0.030303030303030304, p=0.0027594561852200836
        let r = fisher_exact_2x2(1, 9, 11, 3);
        close(r.odds_ratio, 0.030_303_030_303_030_304, 1e-12);
        close(r.p, 0.002_759_456_185_220_083_6, 1e-12);
    }

    #[test]
    fn symmetric_table_is_one() {
        let r = fisher_exact_2x2(10, 10, 10, 10);
        close(r.odds_ratio, 1.0, 1e-12);
        assert_eq!(r.p, 1.0);
    }

    #[test]
    fn strong_association_tiny_p() {
        // [[100,2],[3,50]] -> scipy or=833.3333333333334, p=2.014876734090061e-34
        let r = fisher_exact_2x2(100, 2, 3, 50);
        close(r.odds_ratio, 833.333_333_333_333_4, 1e-12);
        close(r.p, 2.014_876_734_090_061e-34, 1e-9);
    }

    #[test]
    fn zero_cell_infinite_odds() {
        // [[5,0],[3,7]] : b == 0 -> odds ratio infinite, finite p.
        let r = fisher_exact_2x2(5, 0, 3, 7);
        assert!(r.odds_ratio.is_infinite());
        assert!(r.p > 0.0 && r.p <= 1.0);
    }

    #[test]
    fn zero_margin_is_nan_p_one() {
        // First column total a+c == 0.
        let r = fisher_exact_2x2(0, 5, 0, 7);
        assert!(r.odds_ratio.is_nan());
        assert_eq!(r.p, 1.0);
    }
}
