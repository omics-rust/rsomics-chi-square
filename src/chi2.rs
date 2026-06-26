//! Chi-square test of independence on a contingency table, matching
//! `scipy.stats.chi2_contingency`.
//!
//! Expected frequencies are E_ij = (row_sum_i · col_sum_j) / N. The Pearson
//! statistic is Σ(O−E)²/E; with `lambda_="log-likelihood"` (the G-test) it is
//! 2·Σ O·ln(O/E). Degrees of freedom are (r−1)(c−1), and the p-value is the
//! chi-squared survival function at that df.
//!
//! Yates' continuity correction follows SciPy exactly: it applies only when
//! df == 1, adjusting each observed cell toward its expectation by
//! min(0.5, |E−O|) in the direction of E. This is gentler than the textbook
//! "subtract 0.5 from |O−E|" when a cell already lies within 0.5 of E.

use serde::Serialize;

use crate::igamc::chi2_sf;
use rsomics_common::{Result, RsomicsError};

/// Which divergence statistic to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Statistic {
    /// Pearson chi-square, Σ(O−E)²/E.
    Pearson,
    /// G-test / log-likelihood ratio, 2·Σ O·ln(O/E).
    GTest,
}

/// Result of a chi-square (or G-test) test of independence.
#[derive(Debug, Clone, Serialize)]
pub struct ChiSquareResult {
    /// Test statistic (Pearson chi-square or G), chi-squared distributed under H₀.
    pub chi2: f64,
    /// Degrees of freedom, (r−1)(c−1).
    pub df: usize,
    /// Survival-function p-value, chi2.sf(statistic, df).
    pub p: f64,
}

/// A contingency table of non-negative integer counts, stored row-major.
#[derive(Debug, Clone)]
pub struct Table {
    pub rows: usize,
    pub cols: usize,
    pub counts: Vec<f64>,
}

impl Table {
    /// Build a table from `rows × cols` counts in row-major order.
    pub fn new(rows: usize, cols: usize, counts: Vec<f64>) -> Result<Self> {
        if rows < 1 || cols < 1 {
            return Err(RsomicsError::InvalidInput(
                "contingency table needs at least one row and one column".into(),
            ));
        }
        if counts.len() != rows * cols {
            return Err(RsomicsError::InvalidInput(format!(
                "table has {} cells but {rows}×{cols} = {} were declared",
                counts.len(),
                rows * cols
            )));
        }
        Ok(Self { rows, cols, counts })
    }
}

/// Run the chi-square test of independence on `table`.
///
/// `correction` enables Yates' continuity correction (only effective when
/// df == 1, matching SciPy). `statistic` selects Pearson or the G-test.
pub fn chi2_contingency(
    table: &Table,
    correction: bool,
    statistic: Statistic,
) -> Result<ChiSquareResult> {
    let (rows, cols) = (table.rows, table.cols);

    let mut row_sums = vec![0.0_f64; rows];
    let mut col_sums = vec![0.0_f64; cols];
    let mut total = 0.0_f64;
    for (r, row) in table.counts.chunks_exact(cols).enumerate() {
        for (c, &o) in row.iter().enumerate() {
            row_sums[r] += o;
            col_sums[c] += o;
            total += o;
        }
    }
    if total == 0.0 {
        return Err(RsomicsError::InvalidInput(
            "contingency table is all zeros".into(),
        ));
    }

    // E_ij = row_i · col_j / N. A zero in any margin yields a zero expectation,
    // which SciPy rejects rather than dividing by it.
    for (r, &rs) in row_sums.iter().enumerate() {
        for (c, &cs) in col_sums.iter().enumerate() {
            if rs * cs / total == 0.0 {
                return Err(RsomicsError::InvalidInput(format!(
                    "expected frequency is zero at row {r}, col {c} (a margin sums to zero)"
                )));
            }
        }
    }

    let df = (rows - 1) * (cols - 1);
    if df == 0 {
        // Degenerate table (a single row or column): O == E everywhere.
        return Ok(ChiSquareResult {
            chi2: 0.0,
            df,
            p: 1.0,
        });
    }

    let apply_yates = correction && df == 1;
    let mut stat = 0.0_f64;
    for (row, &rs) in table.counts.chunks_exact(cols).zip(&row_sums) {
        for (&cell, &cs) in row.iter().zip(&col_sums) {
            let e = rs * cs / total;
            let mut o = cell;
            if apply_yates {
                let diff = e - o;
                let magnitude = diff.abs().min(0.5);
                o += magnitude * diff.signum();
            }
            stat += match statistic {
                Statistic::Pearson => {
                    let d = o - e;
                    d * d / e
                }
                // xlogy(o, o/e): the term is 0 when o == 0.
                Statistic::GTest => {
                    if o == 0.0 {
                        0.0
                    } else {
                        2.0 * o * (o / e).ln()
                    }
                }
            };
        }
    }

    let p = chi2_sf(stat, df as f64);
    Ok(ChiSquareResult { chi2: stat, df, p })
}

#[cfg(test)]
mod tests {
    use super::{Statistic, Table, chi2_contingency};

    fn close(got: f64, want: f64, rel: f64) {
        let d = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(d <= rel, "got {got} want {want} rel {d:e} > {rel:e}");
    }

    fn table(rows: usize, cols: usize, v: &[f64]) -> Table {
        Table::new(rows, cols, v.to_vec()).unwrap()
    }

    #[test]
    fn pearson_2x2_with_yates() {
        let t = table(2, 2, &[10.0, 20.0, 30.0, 40.0]);
        let r = chi2_contingency(&t, true, Statistic::Pearson).unwrap();
        close(r.chi2, 0.446_428_571_428_571_4, 1e-12);
        assert_eq!(r.df, 1);
        close(r.p, 0.504_035_866_452_504_6, 1e-12);
    }

    #[test]
    fn pearson_2x2_no_correction() {
        let t = table(2, 2, &[10.0, 20.0, 30.0, 40.0]);
        let r = chi2_contingency(&t, false, Statistic::Pearson).unwrap();
        close(r.chi2, 0.793_650_793_650_793_6, 1e-12);
        close(r.p, 0.372_998_483_613_486_86, 1e-12);
    }

    #[test]
    fn yates_only_applies_at_df_one() {
        // A 3x3 table has df == 4; correction must be a no-op there.
        let t = table(3, 3, &[10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 5.0, 15.0, 25.0]);
        let with = chi2_contingency(&t, true, Statistic::Pearson).unwrap();
        let without = chi2_contingency(&t, false, Statistic::Pearson).unwrap();
        assert_eq!(with.chi2, without.chi2);
        close(with.chi2, 9.088_319_088_319_091, 1e-12);
        assert_eq!(with.df, 4);
        close(with.p, 0.058_929_438_709_323_78, 1e-12);
    }

    #[test]
    fn pearson_2x3() {
        let t = table(2, 3, &[12.0, 5.0, 29.0, 7.0, 33.0, 10.0]);
        let r = chi2_contingency(&t, true, Statistic::Pearson).unwrap();
        close(r.chi2, 31.091_089_596_901_95, 1e-12);
        assert_eq!(r.df, 2);
        close(r.p, 1.772_783_397_693_199_7e-7, 1e-12);
    }

    #[test]
    fn gtest_log_likelihood() {
        // G-test reported without continuity correction, as SciPy does.
        let t = table(2, 2, &[1.0, 9.0, 11.0, 3.0]);
        let r = chi2_contingency(&t, false, Statistic::GTest).unwrap();
        close(r.chi2, 12.221_169_703_393_98, 1e-12);
        close(r.p, 0.000_472_502_980_875_963_86, 1e-12);
    }

    #[test]
    fn gtest_3x3() {
        let t = table(3, 3, &[10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 5.0, 15.0, 25.0]);
        let r = chi2_contingency(&t, false, Statistic::GTest).unwrap();
        close(r.chi2, 9.777_367_846_067_989, 1e-12);
        close(r.p, 0.044_349_608_714_103_025, 1e-12);
    }

    #[test]
    fn single_row_is_degenerate() {
        let t = table(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let r = chi2_contingency(&t, true, Statistic::Pearson).unwrap();
        assert_eq!(r.chi2, 0.0);
        assert_eq!(r.df, 0);
        assert_eq!(r.p, 1.0);
    }

    #[test]
    fn zero_margin_is_rejected() {
        let t = table(2, 2, &[0.0, 0.0, 5.0, 7.0]);
        assert!(chi2_contingency(&t, true, Statistic::Pearson).is_err());
    }
}
