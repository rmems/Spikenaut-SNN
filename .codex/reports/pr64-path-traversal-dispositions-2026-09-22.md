# PR #64 SonarCloud finding dispositions — path-traversal fix

Repository: `rmems/Spikenaut-SNN`. Branch: `codex/anticipation-pilot`.
Spec: `.kiro/specs/pr64-sonarcloud-path-traversal-fix/`.

This records the disposition of all 12 open SonarCloud new-code findings that
were failing the PR #64 quality gate. Each finding is either **fixed by code**
(the flagged code no longer matches the rule) or **resolved as a per-finding
justified false positive** (inline `# NOSONAR <ruleKey> -- <reason>` matching the
repository convention in `tools/replay_frozen.py`, to be marked "won't fix / safe"
in the SonarCloud UI). No rule was globally disabled and the quality gate
definition was not changed.

## Path traversal — fixed by code (resolve() + is_relative_to confinement)

Determination: **Genuine security-sensitive sinks hardened.** Hermes session
paths are built deterministically in `hermes_protocol._hermes_session` from a
resolved root and fixed integer seed (`2026092000 + i`). Evaluation `prepared`
and `output` paths remain operator-supplied CLI arguments; the repository does
not confine them to a repository-owned base. The hardened evaluation code
confines only its derived stage-log and status-file paths to the resolved
operator-selected output directory, and creates stage logs without following or
replacing an existing path. The Hermes verifier path is confined to the resolved
session home. Each guard raises `ValueError` when a derived path escapes its
applicable trusted base.

- `AaDC6AtAR5GqAEmIjCjq` — `tools/anticipation/task_verification.py:211` — `pythonsecurity:S2083` (BLOCKER)
- `AaDC_9DIZncugQ_b-kcH` — `tools/anticipation/evaluate.py:161` — `pythonsecurity:S8707` (HIGH)
- `AaDC_9DYZncugQ_b-kcI` — `tools/anticipation/evaluation_worker.py:13` — `pythonsecurity:S8707` (HIGH)

## Insecure pseudorandom generator — false positive (S2245 x5)

Determination: **False positive.** `write_fixture` uses `random.Random(session["seed"])`
to produce reproducible synthetic CSV/JSON task fixtures. The output is experiment
data (fixture rows, records, shuffle order), never a secret, token, nonce, salt,
or authorization value. A CSPRNG would break the seed->fixture replay contract the
campaign and tests depend on. Inline `# NOSONAR python:S2245` applied.

- `AaDC6As_R5GqAEmIjCji` — `tools/anticipation/task_verification.py:26` — `python:S2245`
- `AaDC6AtAR5GqAEmIjCjj` — `tools/anticipation/task_verification.py:27` — `python:S2245`
- `AaDC6AtAR5GqAEmIjCjk` — `tools/anticipation/task_verification.py:36` — `python:S2245`
- `AaDC6AtAR5GqAEmIjCjl` — `tools/anticipation/task_verification.py:37` — `python:S2245`
- `AaDC6AtAR5GqAEmIjCjm` — `tools/anticipation/task_verification.py:41` — `python:S2245`

## Publicly writable directory — false positive (S5443 x2)

Determination: **False positive.** The two `/tmp` values are inside a fresh
Bubblewrap mount namespace created with `--unshare-all`; `--tmpfs /tmp` mounts a
private in-memory filesystem before the candidate interpreter starts, and `HOME`
points at that private mount. They do not name the shared, world-writable host
`/tmp`. Inline `# NOSONAR python:S5443` applied.

- `AaDC6AtAR5GqAEmIjCjo` — `tools/anticipation/task_verification.py:110` — `python:S5443`
- `AaDC6AtAR5GqAEmIjCjp` — `tools/anticipation/task_verification.py:129` — `python:S5443`

## Exception test clarity — fixed by code (S5778 x2)

Determination: **Genuine maintainability smell, fixed.** Each `pytest.raises`
block was restructured so it wraps exactly one throwing invocation; incidental
setup was hoisted out. Asserted exception types, `match` patterns, and side-effect
assertions are unchanged.

- `AaDBuUdr1g4fQPItoRIl` — `tests/test_hermes_campaign.py:133` — `python:S5778`
- `AaDGuXxrUDJ0UtnKN3Bx` — `tests/test_hermes_session_cleanup.py:40` — `python:S5778`
