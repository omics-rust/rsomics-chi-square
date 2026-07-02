//! Fisher's exact test for a 2×2 contingency table, matching
//! `scipy.stats.fisher_exact` with the two-sided alternative.
//!
//! For a table [[a, b], [c, d]] the hypergeometric distribution is
//! parametrised as M = a+b+c+d, n1 = a+b (first row total), n = a+c (first
//! column total), and the cell count a is the random variable. The two-sided
//! p-value sums every hypergeometric pmf no greater than the observed table's
//! pmf. The inclusion test uses a relative tolerance of 1e-7 rather than an
//! absolute tie: the lgamma-based pmf carries ~1e-13 relative error, so a
//! tighter cutoff spuriously drops the opposite-tail cell that is genuinely
//! equal to pmf(observed) on near-symmetric tables, undercounting the tail and
//! breaking reflection invariance. The 1e-7 window admits true ties while the
//! spacing between distinct hypergeometric pmfs keeps genuine non-ties out.
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
    let n = a + c;
    let m = n1 + c + d;

    let lo = (n - (m - n1)).max(0);
    let hi = n1.min(n);

    let pexact = hypergeom_pmf(a, m, n1, n);
    let tol = pexact * (1.0 + 1e-7);

    let mut p = 0.0;
    let mut k = lo;
    while k <= hi {
        let pk = hypergeom_pmf(k, m, n1, n);
        if pk <= tol {
            p += pk;
        }
        k += 1;
    }

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
        close(r.p, 1.0, 1e-12);
    }

    // Near-symmetric tables where the opposite-tail cell equals pmf(observed) to
    // within lgamma rounding. A tie-inclusion cutoff tighter than the pmf's own
    // error drops that cell and undercounts the tail; these lock in the scipy
    // 1.17.1 fisher_exact two-sided values.
    #[test]
    fn near_symmetric_reflection_invariant() {
        let cases = [
            (10, 16, 25, 19, 0.215_853_063_510_954_62),
            (29, 39, 24, 14, 0.067_657_451_865_430_01),
            (32, 16, 8, 24, 0.000_518_578_810_011_806_9),
            (23, 6, 28, 6, 1.0),
        ];
        for (a, b, c, d, want) in cases {
            let r = fisher_exact_2x2(a, b, c, d);
            close(r.p, want, 1e-9);
            // Reflection invariance: swapping rows must not change the p-value.
            let s = fisher_exact_2x2(c, d, a, b);
            close(s.p, want, 1e-9);
        }
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
