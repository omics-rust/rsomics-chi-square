//! Value-exact compatibility against `scipy.stats.chi2_contingency` and
//! `scipy.stats.fisher_exact`.
//!
//! `tests/golden/expected.tsv` holds reference statistics computed once with
//! SciPy 1.17.1; this test reruns the rsomics-chi-square binary on the same
//! committed tables and asserts every field matches without invoking SciPy.

use std::path::{Path, PathBuf};
use std::process::Command;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-chi-square"))
}

fn run(args: &[&str]) -> Vec<f64> {
    let table = golden_dir().join(args[0]);
    let mut full = vec![table.to_str().unwrap()];
    full.extend_from_slice(&args[1..]);
    let out = Command::new(bin())
        .args(&full)
        .output()
        .expect("binary runs");
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .split('\t')
        .map(|s| s.parse::<f64>().expect("numeric output field"))
        .collect()
}

fn close(got: f64, want: f64, rel: f64) {
    if want == 0.0 {
        assert!(got.abs() <= rel, "got {got} want {want}");
        return;
    }
    let d = (got - want).abs() / want.abs();
    assert!(d <= rel, "got {got:e} want {want:e} rel {d:e} > {rel:e}");
}

struct Expected {
    mode: String,
    name: String,
    fields: Vec<f64>,
}

fn load_expected() -> Vec<Expected> {
    let raw = std::fs::read_to_string(golden_dir().join("expected.tsv")).unwrap();
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            Expected {
                mode: cols[0].to_string(),
                name: cols[1].to_string(),
                fields: cols[2..].iter().map(|s| s.parse().unwrap()).collect(),
            }
        })
        .collect()
}

#[test]
fn matches_scipy_value_exact() {
    for e in load_expected() {
        let tsv = format!("{}.tsv", e.name);
        match e.mode.as_str() {
            "pearson" => {
                let got = run(&[&tsv]);
                close(got[0], e.fields[0], 1e-12);
                assert_eq!(got[1] as usize, e.fields[1] as usize, "df {}", e.name);
                close(got[2], e.fields[2], 1e-12);
            }
            "pearson_nc" => {
                let got = run(&[&tsv, "--no-correction"]);
                close(got[0], e.fields[0], 1e-12);
                assert_eq!(got[1] as usize, e.fields[1] as usize, "df {}", e.name);
                close(got[2], e.fields[2], 1e-12);
            }
            "gtest" => {
                let got = run(&[&tsv, "--no-correction", "--gtest"]);
                close(got[0], e.fields[0], 1e-12);
                assert_eq!(got[1] as usize, e.fields[1] as usize, "df {}", e.name);
                close(got[2], e.fields[2], 1e-12);
            }
            "fisher" => {
                let got = run(&[&tsv, "--exact"]);
                close(got[0], e.fields[0], 1e-12);
                close(got[1], e.fields[1], 1e-9);
            }
            other => panic!("unknown mode {other}"),
        }
    }
}
