// SPDX-License-Identifier: MIT OR Apache-2.0

//! Smoke test for the `neuromod` integration (issue #5): a host-side
//! [`HostLif`] must construct a real `neuromod::LifNeuron`, step it, and
//! resolve from crates.io. Removing the crate fails this file and the
//! lockfile assertion.

use std::path::Path;

use neuromod::LifNeuron;
use spikenaut_snn::HostLif;

/// The adapter hands back the published type, not a local stand-in.
#[test]
fn host_lif_exposes_the_published_lif_neuron() {
    let cell = HostLif::new();
    let published = LifNeuron::new();
    assert_eq!(cell.inner().threshold, published.threshold);
    assert_eq!(cell.inner().decay_rate, published.decay_rate);
    assert_eq!(
        cell.inner().membrane_potential,
        published.membrane_potential
    );
}

/// `Default` is the published default cell, and a resting cell never fires.
#[test]
fn a_default_cell_rests_without_firing() {
    let mut cell = HostLif::default();
    let published = LifNeuron::new();
    assert_eq!(cell.inner().threshold, published.threshold);
    assert_eq!(cell.inner().decay_rate, published.decay_rate);
    assert_eq!(cell.membrane_potential(), published.membrane_potential);

    for tick in 0..8 {
        cell.integrate(0.0);
        assert!(cell.check_fire().is_none(), "rest fired at tick {tick}");
    }
    assert_eq!(cell.membrane_potential(), published.membrane_potential);
}

/// A quiet subthreshold input leaks and does not fire.
#[test]
fn a_subthreshold_step_leaks_without_firing() {
    let mut cell = HostLif::new();
    let threshold = cell.inner().threshold;
    cell.integrate(threshold * 0.1);
    let after = cell.membrane_potential();
    assert!(after.is_finite());
    assert!(after > 0.0, "a small pulse must raise the membrane");
    assert!(after < threshold);
    assert!(cell.check_fire().is_none());
}

/// Acceptance from issue #5: the direct `neuromod` dependency resolves to
/// 0.7.x from crates.io, not from a git or sibling-path pin. The `"0.7"`
/// caret is `>=0.7.0, <0.8.0`, and plasticity-lab 0.3 shares that one copy —
/// the neuromod 0.6 alias is gone.
#[test]
fn neuromod_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");

    let entries: Vec<&str> = lock
        .split("[[package]]")
        .filter(|block| block.contains(r#"name = "neuromod""#))
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "the neuromod 0.6 alias is gone; exactly one neuromod may resolve, got {entries:?}"
    );

    let neuromod_07 = entries[0];
    assert!(
        neuromod_07.contains(r#"version = "0.7.0""#),
        "Cargo.lock must lock neuromod 0.7.0, got:\n{neuromod_07}",
    );
    assert!(
        neuromod_07.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "neuromod 0.7.0 must come from the crates.io registry, got:\n{neuromod_07}",
    );

    let root_pkg = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "spikenaut-snn""#))
        .expect("Cargo.lock has a spikenaut-snn package entry");
    assert!(
        root_pkg.lines().any(|line| line.trim() == "\"neuromod\","),
        "spikenaut-snn must depend on neuromod, got:\n{root_pkg}",
    );

    assert!(
        !lock.contains("source = \"git+") && !lock.contains("[[patch"),
        "every locked package must come from the registry",
    );
}
