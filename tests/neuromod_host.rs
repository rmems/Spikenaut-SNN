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

/// Acceptance from issue #5: `neuromod` resolves to 0.5.x from crates.io,
/// not from a git or sibling-path pin. The `"0.5"` caret is
/// `>=0.5.0, <0.6.0`; a later 0.5.x lockfile bump must still pass.
#[test]
fn neuromod_resolves_from_crates_io() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");

    let entry = lock
        .split("[[package]]")
        .find(|block| block.contains(r#"name = "neuromod""#))
        .expect("Cargo.lock has a neuromod package entry");

    assert!(
        entry.contains(r#"source = "registry+https://github.com/rust-lang/crates.io-index""#),
        "neuromod must come from the crates.io registry, got:\n{entry}",
    );
    // Same series check as axon-encoder's `"0.4"` caret: `version = "0.5.`
    // matches 0.5.2 and 0.5.10, not 0.6.0.
    assert!(
        entry.contains(r#"version = "0.5."#),
        "neuromod must resolve to 0.5.x (>=0.5, <0.6), got:\n{entry}",
    );
    assert!(
        !lock.contains("source = \"git+") && !lock.contains("[[patch"),
        "every locked package must come from the registry",
    );
}
