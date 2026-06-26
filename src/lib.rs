//! Chi-square test of independence on a contingency table —
//! `scipy.stats.chi2_contingency` equivalent, with a G-test variant and a
//! 2×2 Fisher exact test (`scipy.stats.fisher_exact`).
//!
//! Input is a TSV of integer counts: one table row per line, columns separated
//! by tabs. The Pearson chi-square (or, under `--gtest`, the log-likelihood
//! G-test) reports `chi2`, `df`, and the survival-function p-value. Under
//! `--exact` the table must be 2×2 and the Fisher exact odds ratio + two-sided
//! p-value are reported instead.

mod chi2;
mod fisher;
mod igamc;

use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

pub use chi2::{ChiSquareResult, Statistic, Table, chi2_contingency};
pub use fisher::{FisherResult, fisher_exact_2x2};

/// Parse a whitespace/tab-delimited grid of non-negative integer counts into a
/// rectangular [`Table`]. Blank lines are skipped; every non-blank line must
/// have the same column count.
pub fn parse_table<R: BufRead>(reader: R) -> Result<Table> {
    let mut counts: Vec<f64> = Vec::new();
    let mut cols: Option<usize> = None;
    let mut rows = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        let fields: Vec<&str> = line.split('\t').filter(|s| !s.trim().is_empty()).collect();
        if fields.is_empty() {
            continue;
        }
        match cols {
            None => cols = Some(fields.len()),
            Some(width) if width != fields.len() => {
                return Err(RsomicsError::InvalidInput(format!(
                    "line {}: row has {} columns but the table is {width} wide",
                    lineno + 1,
                    fields.len()
                )));
            }
            Some(_) => {}
        }
        for f in fields {
            let v: i64 = f.trim().parse().map_err(|_| {
                RsomicsError::InvalidInput(format!(
                    "line {}: count '{f}' is not an integer",
                    lineno + 1
                ))
            })?;
            if v < 0 {
                return Err(RsomicsError::InvalidInput(format!(
                    "line {}: count '{f}' is negative",
                    lineno + 1
                )));
            }
            counts.push(v as f64);
        }
        rows += 1;
    }

    let cols = cols.ok_or_else(|| RsomicsError::InvalidInput("empty contingency table".into()))?;
    Table::new(rows, cols, counts)
}

/// Run a Fisher exact test on a table that must be exactly 2×2.
pub fn fisher_from_table(table: &Table) -> Result<FisherResult> {
    if table.rows != 2 || table.cols != 2 {
        return Err(RsomicsError::InvalidInput(format!(
            "--exact requires a 2×2 table, got {}×{}",
            table.rows, table.cols
        )));
    }
    let a = table.counts[0] as i64;
    let b = table.counts[1] as i64;
    let c = table.counts[2] as i64;
    let d = table.counts[3] as i64;
    Ok(fisher_exact_2x2(a, b, c, d))
}

#[cfg(test)]
mod tests {
    use super::{Statistic, chi2_contingency, fisher_from_table, parse_table};

    #[test]
    fn parses_rectangular_grid() {
        let t = parse_table("10\t20\n30\t40\n".as_bytes()).unwrap();
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 2);
        assert_eq!(t.counts, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn skips_blank_lines() {
        let t = parse_table("1\t2\n\n3\t4\n".as_bytes()).unwrap();
        assert_eq!(t.rows, 2);
    }

    #[test]
    fn rejects_ragged_rows() {
        assert!(parse_table("1\t2\n3\t4\t5\n".as_bytes()).is_err());
    }

    #[test]
    fn rejects_non_integer() {
        assert!(parse_table("1\t2.5\n3\t4\n".as_bytes()).is_err());
    }

    #[test]
    fn rejects_negative() {
        assert!(parse_table("1\t-2\n3\t4\n".as_bytes()).is_err());
    }

    #[test]
    fn end_to_end_chi2() {
        let t = parse_table("10\t20\n30\t40\n".as_bytes()).unwrap();
        let r = chi2_contingency(&t, true, Statistic::Pearson).unwrap();
        assert!((r.chi2 - 0.446_428_571_428_571_4).abs() < 1e-12);
    }

    #[test]
    fn fisher_requires_2x2() {
        let t = parse_table("1\t2\t3\n4\t5\t6\n".as_bytes()).unwrap();
        assert!(fisher_from_table(&t).is_err());
    }
}
