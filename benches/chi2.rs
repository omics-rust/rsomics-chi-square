use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_chi_square::{Statistic, Table, chi2_contingency};
use std::hint::black_box;

fn synth_table(rows: usize, cols: usize) -> Table {
    let mut counts = Vec::with_capacity(rows * cols);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..rows * cols {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        counts.push(((state % 1000) + 1) as f64);
    }
    Table::new(rows, cols, counts).unwrap()
}

fn bench_chi2(c: &mut Criterion) {
    let big = synth_table(3000, 3000);
    c.bench_function("chi2_3000x3000_pearson", |b| {
        b.iter(|| {
            let r = chi2_contingency(black_box(&big), false, Statistic::Pearson).unwrap();
            black_box(r.chi2)
        });
    });
    c.bench_function("chi2_3000x3000_gtest", |b| {
        b.iter(|| {
            let r = chi2_contingency(black_box(&big), false, Statistic::GTest).unwrap();
            black_box(r.chi2)
        });
    });
}

criterion_group!(benches, bench_chi2);
criterion_main!(benches);
