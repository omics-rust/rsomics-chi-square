use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_chi_square::{Statistic, Table, chi2_contingency, fisher_from_table, parse_table};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Chi-square test of independence on a contingency table
/// (`scipy.stats.chi2_contingency`).
///
/// Input is a TSV grid of non-negative integer counts: one table row per line,
/// columns tab-separated. Output is `chi2<TAB>df<TAB>p`. With `--exact` the
/// table must be 2×2 and a Fisher exact test reports `oddsratio<TAB>p` instead.
#[derive(Parser, Debug)]
#[command(name = "rsomics-chi-square", version, about, long_about = None)]
pub struct Cli {
    /// Contingency table TSV (integer counts); `-` or omitted reads stdin.
    #[arg(value_name = "TABLE")]
    pub table: Option<PathBuf>,

    /// Disable Yates' continuity correction (it only affects 2×2 / df==1 tables).
    #[arg(long = "no-correction")]
    pub no_correction: bool,

    /// Use the G-test (log-likelihood ratio) statistic instead of Pearson.
    #[arg(long = "gtest", conflicts_with = "exact")]
    pub gtest: bool,

    /// Fisher's exact test for a 2×2 table (reports oddsratio and two-sided p).
    #[arg(long = "exact")]
    pub exact: bool,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Report {
    ChiSquare {
        chi2: f64,
        df: usize,
        p: f64,
    },
    Fisher {
        #[serde(rename = "oddsratio")]
        odds_ratio: f64,
        p: f64,
    },
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let table = read_table(self.table.as_ref())?;

            let report = if self.exact {
                let f = fisher_from_table(&table)?;
                if !common.json {
                    println!("{}\t{}", f.odds_ratio, f.p);
                }
                Report::Fisher {
                    odds_ratio: f.odds_ratio,
                    p: f.p,
                }
            } else {
                let statistic = if self.gtest {
                    Statistic::GTest
                } else {
                    Statistic::Pearson
                };
                let r = chi2_contingency(&table, !self.no_correction, statistic)?;
                if !common.json {
                    println!("{}\t{}\t{}", r.chi2, r.df, r.p);
                }
                Report::ChiSquare {
                    chi2: r.chi2,
                    df: r.df,
                    p: r.p,
                }
            };
            Ok(report)
        })
    }
}

fn read_table(path: Option<&PathBuf>) -> Result<Table, RsomicsError> {
    match path {
        Some(p) if p.as_os_str() != "-" => {
            let f = File::open(p).map_err(RsomicsError::Io)?;
            parse_table(BufReader::new(f))
        }
        _ => {
            let stdin = io::stdin();
            parse_table(stdin.lock())
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
