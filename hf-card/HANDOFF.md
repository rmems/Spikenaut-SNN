# Hugging Face model-card payload — handoff

Everything in this directory is destined for the **Hugging Face** repo
[`rmems/Spikenaut-SNN`](https://huggingface.co/rmems/Spikenaut-SNN), not for this GitHub repo. It is
staged here only so a local session can pick it up with a `git pull` instead of files being passed
by hand. Nothing here is imported or referenced by the crate or the verifier.

## Why

The HF model card doubles as this repo's `README.md`, but the HF copy was last updated 2026-07-25
and never received the rewrite that landed in #22 and the PRs after it. The live card still tells
roughly 500 people who have downloaded these weights two things that are not true:

| Live HF card | Reality |
|---|---|
| "Channels 14-15 … the SNN receives negative reward and **learns to avoid states that could damage the hardware**." | Design intent only. No training run links the thermal channels to the shipped weights; there is no reward signal and no online update. |
| Output weights — "**Real trained weights**" | An unseeded `torch.randn(3, 16) * 0.1`, exported with no training step. Reading those three rows evaluates a random readout. |

## File mapping

Two files are stored under `dot-` names so git does not apply them to *this* repo. Rename on upload:

| Here | Lands on HF as |
|---|---|
| `README.md` | `README.md` |
| `dot-gitattributes` | `.gitattributes` |
| `dot-gitignore` | `.gitignore` |
| `LICENSE-MIT` | `LICENSE-MIT` |
| `LICENSE-APACHE` | `LICENSE-APACHE` |
| `assets/architecture.svg` | `assets/architecture.svg` |
| `assets/architecture.png` | `assets/architecture.png` |

`.gitattributes` in *this* directory is GitHub-side only — the repo root marks `*.png` as LFS, and
this override keeps the rendered diagram an ordinary blob so a clone without git-lfs gets the real
image. Do not upload it.

**The artifacts are not part of this payload.** `config.json` and all five files under
`dataset/merged_v2/` are already byte-identical on both sides — `parameters_decay.mem` hashes
`6db9fecba2c2411c` in each, the same digest the card's Status section cites as its own evidence.
This is a card-and-metadata change only; do not re-upload weights.

## What changed against the live card

- **Ported** from this repo's `README.md`: Status, the research question, system model, safety
  principle, architecture, inputs, provenance, known limitations, hardware baseline, roadmap.
- **Adapted**, because a verbatim copy would trade old false claims for new ones — the live HF repo
  has no `tools/`, no `src/`, no `Cargo.toml`:
  - `## Files` describes the HF file set and attributes the verifier and crate to GitHub.
  - `## Ecosystem` keeps the component table and its peer-process distinction, but its status column
    no longer claims a `Cargo.toml` that is not in the download.
  - The channels 14-15 paragraph no longer says "the repository now carries code" — on HF it does not.
  - A scoping note under Status records that provenance citations (deleted scripts, commit hashes)
    refer to GitHub's history, and that the artifacts are identical.
- **Added**: a `### Loading in Python` section, a `datasets:` cross-link to
  `rmems/Spikenaut-SNN-Telemetry`, three tags, and `assets/architecture.png`.
- **`## The Story` is byte-identical to the live card**, all four paragraphs, including a trailing
  space. The sentence "This model is the brain -- the trained neural weights that turn raw telemetry
  into decisions" was dropped by the #22 rewrite and **restored at the author's explicit request**.
  It sits alongside a Status table that says the weights are not established as trained; that
  juxtaposition is intentional and is the author's call. Do not re-remove it.

## Verification before publishing

1. Front matter parses as YAML and keeps `model_name`, `pipeline_tag`, `license`, `tags`, `datasets`.
2. No path claim without attribution — grep for `tools/`, `src/`, `Cargo.toml`, `.rs`; the two
   surviving `tools/verify_q88.py` mentions are both explicitly attributed to the GitHub repo.
3. `assets/architecture.png` resolves from the card.
4. `## The Story` still diffs clean against the live card.
5. The `### Loading in Python` snippet runs against `dataset/merged_v2/` and prints exactly the
   figures the card quotes:

```
thresholds 16: 1.125 .. 1.59375
decays     16: 0.796875 .. 0.94921875
weights    256: 0.75 .. 1.04296875
outputs    48: min -0.1640625 max 0.2578125
matrix     16x16, all positive: True
```

If those differ, the card and the artifacts have diverged — stop and investigate rather than
publishing.

## Publishing

Open it as a **pull request** on the Hub, not a direct commit to `main`; the `hf` CLI upload command
takes a `--create-pr` flag. Confirm the exact invocation against the installed CLI version first —
the command name changed from `huggingface-cli` to `hf`, and flags differ across versions.

## Regenerating the diagram

`assets/architecture.svg` is the source; the PNG is rendered from it. The SVG has an intrinsic size
of 900x560, but Chromium clips roughly the bottom 85px when the viewport matches exactly, which
silently drops the footer — render into a taller viewport (900x660 works) with a page background of
`#fbfbfa` so the extra space is invisible, then check the footer line is present before shipping.
