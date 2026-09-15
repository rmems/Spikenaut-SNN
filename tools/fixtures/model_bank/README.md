# Model-bank manifest (schema_version 1)

A model bank is a directory that contains `model_bank.json` plus the checkpoint
files the manifest names. Linear RM-1327: every selectable checkpoint must be
bound to the metadata that explains its input map and output semantics.

## Stable serialization

`tools.model_bank.dumps_manifest` is the contract. The golden document is
`valid/model_bank.json`. A file is stable when it equals that function's output
byte-for-byte:

* UTF-8, LF newlines, exactly one trailing newline
* `json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"`
* object keys sorted recursively; array order (the `models` list) is significant
* no extra top-level or per-entry keys

Reproduce from the repo root:

```python
from pathlib import Path
from tools.model_bank import dumps_manifest, load_model_bank
raw = Path("tools/fixtures/model_bank/valid/model_bank.json").read_text()
bank = load_model_bank("tools/fixtures/model_bank/valid/model_bank.json")
assert raw == dumps_manifest(__import__("json").loads(raw))
print(bank.select("fixture-ok").checkpoint_digest)
```

## Digests and paths

`checkpoint_digest` is `sha256:` plus 64 lowercase hex characters of the
checkpoint **file bytes**. The path is not hashed. Checkpoint paths are POSIX
and relative to the directory that contains the manifest; they must not be
absolute or contain `..`.

Copying the whole bundle to a new directory without changing bytes still
attests. That is the path-independence rule.

## Required entry fields

| Field | Role |
|---|---|
| `id` | unique within the bank |
| `checkpoint` | relative path to the checkpoint file |
| `checkpoint_digest` | content digest of that file |
| `feature_map_id` | input-map identifier (`spikenaut.feature-map.live-exp-025.v1` for the live bank) |
| `output_contract_id` | output-row / action-map identifier. The shipped value `spikenaut.output-contract.supervisor-v3.rm-1150` **names** the RM-1150 decision contract; this loader does not implement that decision. |
| `numeric_format` | e.g. `q8.8-fixed-point` |
| `training_dataset_digest` | optional, same `sha256:` form |

`schema_version` must be the integer `1`.

## Fixtures

| Bundle | Verdict |
|---|---|
| `valid/` | attests |
| `tampered-checkpoint/` | FAIL `checkpoint_digest` (file bytes changed, declared digest did not) |
| `unsupported-version/` | FAIL `schema_version` |

## Compatibility

`SnnModel::load_default` and `tools/verify_q88.py` remain the single-checkpoint
path for `dataset/merged_v2/`. `wrap_legacy_checkpoint` and
`load_unattested_checkpoint` wrap that layout. Prefer `load_model_bank` when
selecting a model.
