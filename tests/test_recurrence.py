"""Recurrent-variant tests (GH #3 / exp-026).

Run::

    python3 -m unittest tests.test_recurrence -v
    pytest -q tests/test_recurrence.py

Standard library only; pytest is an optional runner.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.hamming_banks import bank_from_model
from tools.hamming_const import N_EXC, N_NEURONS, f32
from tools.hamming_encode import load_jsonl, select_samples
from tools.hamming_lif import LifBank, keep_lif_step
from tools.model_bank import load_model_bank
from tools.q88_core import (
    ParseError,
    decode_q88,
    encode_q88,
    encode_q88_hex,
    load_model,
)
from tools.recurrence_bundle import UNCHANGED_MEM_IMAGES
from tools.recurrence_core import (
    ReadoutScorer,
    Span,
    activity,
    constant_predictor,
    decision_agreement,
    fit_dale_readout,
    fmt_ratio,
    metrics_from_predictions,
    nnls,
    promising,
    readout_target,
    session_spans,
    sign_census,
    step_arm,
    target_argmax_agreement,
)
from tools.recurrence_eval import (
    FIXTURE_JSONL,
    AttestedBase,
    Comparison,
    EvalInput,
    Protocol,
    load_splits,
    prepare_base,
)
from tools.recurrence_eval import main as cli_main
from tools.recurrent_lif import (
    RECURRENCE_KEY,
    RECURRENCE_SCHEMA,
    Recurrence,
    RecurrentState,
    lateral_inhibition,
    recurrence_block,
    recurrence_from_model,
    recurrent_lif_step,
    recurrent_mem_words,
)
from tools.replay_core import (
    REPO_ROOT,
    ReplayConfig,
    build_manifest,
    model_on_numeric_grid,
    replay,
    trace_jsonl,
)
from tools.replay_frozen import main as replay_main
from tools.replay_inputs import load_replay_inputs, select_entry

SHIPPED_DIR = REPO_ROOT / "dataset" / "merged_v2"
SHIPPED_MANIFEST = SHIPPED_DIR / "model_bank.json"
VARIANT_DIR = REPO_ROOT / "dataset" / "merged_v2_rec_v1"
REPLAY_FIXTURE = REPO_ROOT / "tools" / "fixtures" / "replay_frozen" / "telemetry.jsonl"
CONFIG = ReplayConfig(split="all", k=4, i_drive=0.05, missing_policy="encode-zero")


def _grid_model() -> dict:
    return model_on_numeric_grid(
        load_model(SHIPPED_DIR / "snn_model.json"), "q8.8-fixed-point", "snn_model.json"
    )


def _fixture_samples():
    return select_samples(load_jsonl(FIXTURE_JSONL), "all")


def _zero_recurrence() -> Recurrence:
    return Recurrence(
        topology="zero (identity check)",
        weights=tuple(tuple(0.0 for _ in range(N_NEURONS)) for _ in range(N_NEURONS)),
    )


def _write_bank(tmp: Path, model: dict, model_id: str = "rec_test") -> Path:
    payload = (json.dumps(model, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    (tmp / "snn_model.json").write_bytes(payload)
    manifest = tmp / "model_bank.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": model_id,
                        "checkpoint": "snn_model.json",
                        "checkpoint_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        "feature_map_id": "spikenaut.feature-map.live-exp-025.v1",
                        "output_contract_id": "spikenaut.output-contract.supervisor-v3.rm-1150",
                        "numeric_format": "q8.8-fixed-point",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _model_with_recurrence(gain: float = 0.25) -> dict:
    model = load_model(SHIPPED_DIR / "snn_model.json")
    model[RECURRENCE_KEY] = recurrence_block(lateral_inhibition(model, gain))
    return model


def _toy_bank() -> LifBank:
    """Neuron 0 (E) and neuron 12 (I) read axon 0; everything else silent."""
    weights = [[0.0] * N_NEURONS for _ in range(N_NEURONS)]
    weights[0][0] = 1.0
    weights[12][0] = 1.0
    threshold = [10.0] * N_NEURONS
    threshold[0] = 0.5
    threshold[12] = 0.5
    return LifBank(
        name="toy", weights=weights, decay=[0.0] * N_NEURONS,
        threshold=threshold, i_drive=0.0,
    )


def _toy_model() -> dict:
    return {"neurons": [{"inhibitory": i >= N_EXC} for i in range(N_NEURONS)]}


class TestStepContract(unittest.TestCase):
    def test_zero_matrix_is_bit_identical_to_feed_forward(self):
        grid = _grid_model()
        ff = bank_from_model(grid, "ff", 0.05)
        rec = bank_from_model(grid, "rec", 0.05)
        state = RecurrentState()
        zero = _zero_recurrence()
        prev = None
        for sample in _fixture_samples():
            if sample.episode_id != prev:
                ff.reset()
                rec.reset()
                state.reset()
                prev = sample.episode_id
            a = keep_lif_step(ff, list(sample.stim), 4)
            b = recurrent_lif_step(rec, list(sample.stim), 4, zero, state)
            self.assertEqual(a, b)
            self.assertEqual(ff.v, rec.v)

    def test_lateral_inhibition_is_i_to_e_only(self):
        rec = lateral_inhibition(_grid_model(), 0.25)
        self.assertEqual(rec.nonzero(), (N_NEURONS - N_EXC) * N_EXC)
        for post in range(N_NEURONS):
            for pre in range(N_NEURONS):
                expected = -0.25 if (pre >= N_EXC and post < N_EXC) else 0.0
                self.assertEqual(rec.weights[post][pre], expected)

    def test_feedback_is_delayed_one_tick_and_resets_per_session(self):
        rec = lateral_inhibition(_toy_model(), 1.0)
        bank = _toy_bank()
        state = RecurrentState()
        stim = [1.0] + [0.0] * (N_NEURONS - 1)
        # Tick 1: I fires, but its spike only reaches E on the next tick.
        first = recurrent_lif_step(bank, stim, None, rec, state)
        self.assertTrue(first[0] and first[12])
        # Tick 2: last tick's I spike cancels E's input exactly.
        second = recurrent_lif_step(bank, stim, None, rec, state)
        self.assertFalse(second[0])
        self.assertTrue(second[12])
        # Tick 3: still suppressed while I keeps firing.
        self.assertFalse(recurrent_lif_step(bank, stim, None, rec, state)[0])
        # Session boundary: both states reset, E fires again.
        bank.reset()
        state.reset()
        self.assertTrue(recurrent_lif_step(bank, stim, None, rec, state)[0])

    def test_feedback_sums_in_binary32(self):
        rec = lateral_inhibition(_grid_model(), 0.375)
        prev = [i >= N_EXC for i in range(N_NEURONS)]
        fb = rec.feedback(prev)
        self.assertEqual(fb[:N_EXC], [f32(-0.375 * 4)] * N_EXC)
        self.assertEqual(fb[N_EXC:], [0.0] * (N_NEURONS - N_EXC))
        with self.assertRaises(ParseError):
            rec.feedback(prev[:-1])

    def test_feedback_width_refused_by_stepper(self):
        bank = _toy_bank()
        stim = [0.0] * N_NEURONS
        with self.assertRaises(ParseError):
            keep_lif_step(bank, stim, None, feedback=[0.0] * 3)


class TestBlockValidation(unittest.TestCase):
    def _refused(self, mutate, message_part: str):
        model = _model_with_recurrence()
        mutate(model)
        with self.assertRaises(ParseError) as caught:
            recurrence_from_model(model, "ckpt")
        self.assertIn(message_part, str(caught.exception))

    def test_absent_key_is_feed_forward(self):
        self.assertIsNone(recurrence_from_model(_grid_model()))

    def test_round_trip(self):
        rec = lateral_inhibition(_grid_model(), 0.125)
        model = _grid_model()
        model[RECURRENCE_KEY] = recurrence_block(rec)
        self.assertEqual(recurrence_from_model(model), rec)

    def test_null_block_refused(self):
        self._refused(lambda m: m.__setitem__(RECURRENCE_KEY, None), "must be a JSON object")

    def test_unknown_and_missing_keys_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY].__setitem__("gain", 0.25), "unknown key")
        self._refused(lambda m: m[RECURRENCE_KEY].pop("topology"), "missing key")

    def test_schema_and_delay_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY].__setitem__("schema", "v0"), "schema")
        self._refused(lambda m: m[RECURRENCE_KEY].__setitem__("delay_ticks", 2), "delay_ticks")
        self._refused(lambda m: m[RECURRENCE_KEY].__setitem__("delay_ticks", True), "delay_ticks")
        self._refused(lambda m: m[RECURRENCE_KEY].__setitem__("topology", " "), "topology")

    def test_shape_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"].pop(), "15 rows, expected 16")
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][3].pop(), "expected 16")

    def test_off_grid_weight_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][0].__setitem__(12, -0.1), "Q8.8 grid")

    def test_dale_and_diagonal_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][0].__setitem__(12, 0.25), "inhibitory")
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][12].__setitem__(0, -0.25), "excitatory")
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][3].__setitem__(3, 0.25), "self-connection")

    def test_missing_inhibitory_flag_refused(self):
        self._refused(lambda m: m["neurons"][5].pop("inhibitory"), "inhibitory must be a boolean")

    def test_non_numeric_weight_refused(self):
        self._refused(lambda m: m[RECURRENCE_KEY]["weights"][0].__setitem__(12, "nan"), "weights[0][12]")

    def test_lateral_inhibition_gain_rules(self):
        model = _grid_model()
        for bad in (0.0, -0.25, 0.1):
            with self.assertRaises(ParseError):
                lateral_inhibition(model, bad)
        all_e = {"neurons": [{"inhibitory": False} for _ in range(N_NEURONS)]}
        with self.assertRaises(ParseError):
            lateral_inhibition(all_e, 0.25)

    def test_mem_words_are_post_major_q88(self):
        rec = lateral_inhibition(_grid_model(), 0.25)
        words = recurrent_mem_words(rec)
        self.assertEqual(len(words), N_NEURONS * N_NEURONS)
        self.assertEqual(words[0 * N_NEURONS + 12], encode_q88_hex(-0.25))
        self.assertEqual(words[12 * N_NEURONS + 0], "0000")


class TestReplayIntegration(unittest.TestCase):
    def _replay(self, manifest: Path, jsonl: Path = FIXTURE_JSONL):
        bank, entry, samples, sha = load_replay_inputs(manifest, None, jsonl, "all", None)
        result = replay(entry, samples, CONFIG)
        doc = build_manifest(
            entry=entry, bank=bank, jsonl=jsonl, split_manifest_path=None,
            input_sha256=sha, config=CONFIG, result=result,
            trace_bytes=trace_jsonl(result.trace_rows),
        )
        return result, doc

    def test_shipped_bank_records_no_recurrence(self):
        result, doc = self._replay(SHIPPED_MANIFEST)
        self.assertIsNone(result.recurrence)
        self.assertIsNone(doc["stepper"]["recurrence"])

    def test_recurrent_checkpoint_is_replayed_recurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_bank(Path(tmp), _model_with_recurrence(0.25))
            ff, _ = self._replay(SHIPPED_MANIFEST)
            rec, doc = self._replay(manifest)
            again, _ = self._replay(manifest)
        described = doc["stepper"]["recurrence"]
        self.assertEqual(described["schema"], RECURRENCE_SCHEMA)
        self.assertEqual(described["delay_ticks"], 1)
        self.assertEqual(described["nonzero_weights"], 48)
        self.assertEqual(rec.trace_rows, again.trace_rows)
        self.assertNotEqual(
            [row["spikes"] for row in ff.trace_rows],
            [row["spikes"] for row in rec.trace_rows],
        )
        # Reset semantics: the first tick of every session has no feedback.
        prev = None
        for a, b in zip(ff.trace_rows, rec.trace_rows, strict=True):
            if a["session"] != prev:
                self.assertEqual(a["spikes"], b["spikes"])
                prev = a["session"]

    def test_step_arm_matches_replay_trace(self):
        bank, entry, samples, _ = load_replay_inputs(SHIPPED_MANIFEST, None, FIXTURE_JSONL, "all", None)
        result = replay(entry, samples, CONFIG)
        lif = bank_from_model(_grid_model(), "ff", 0.05)
        masks = step_arm(lif, samples, 4, None).masks
        for row, mask in zip(result.trace_rows, masks, strict=True):
            self.assertEqual(row["spikes"], [i for i in range(N_NEURONS) if (mask >> i) & 1])


def _diagonal_fit_case() -> tuple[list[list[float]], list[list[float]], list[bool]]:
    """Diagonal Gram matrix, one shared rhs row, Dale flags as shipped."""
    gram = [[(4.0 if i == j else 0.0) for j in range(N_NEURONS)] for i in range(N_NEURONS)]
    rhs = [[1.0, -1.0, 2.0] for _ in range(N_NEURONS)]
    inhibitory = [i >= N_EXC for i in range(N_NEURONS)]
    return gram, rhs, inhibitory


class TestFitAndMetrics(unittest.TestCase):
    def test_nnls_unconstrained_optimum_when_positive(self):
        u = nnls([[2.0, 0.0], [0.0, 3.0]], [2.0, 3.0])
        self.assertAlmostEqual(u[0], 1.0)
        self.assertAlmostEqual(u[1], 1.0)

    def test_nnls_clamps_negative_coordinate(self):
        u = nnls([[1.0, 0.0], [0.0, 1.0]], [1.0, -1.0])
        self.assertAlmostEqual(u[0], 1.0)
        self.assertEqual(u[1], 0.0)

    def test_nnls_drops_a_passive_coordinate_on_the_way(self):
        # Coordinate 0 enters first (largest gradient, ties to the lowest
        # index), then 2; the passive-set solution is then infeasible in 0,
        # so the inner loop takes the alpha step and drops 0 again.
        a = [[19.0, -3.0, 3.0], [-3.0, 10.0, 0.0], [3.0, 0.0, 3.0]]
        c = [2.0, -4.0, 2.0]
        u = nnls(a, c)
        self.assertEqual(u[:2], [0.0, 0.0])
        self.assertAlmostEqual(u[2], 2 / 3)
        # KKT: zero gradient where u > 0, non-positive where u == 0.
        grad = [c[i] - sum(a[i][j] * u[j] for j in range(3)) for i in range(3)]
        self.assertAlmostEqual(grad[2], 0.0)
        self.assertLessEqual(grad[0], 1e-12)
        self.assertLessEqual(grad[1], 0.0)

    def test_nnls_refuses_singular_and_empty_systems(self):
        with self.assertRaises(ParseError):
            nnls([[0.0]], [1.0])
        with self.assertRaises(ParseError):
            nnls([], [])

    def test_agreement_and_baseline_refuse_empty_traces(self):
        scorer = ReadoutScorer([0.0] * (N_NEURONS * 3))
        with self.assertRaises(ParseError):
            decision_agreement([], scorer, [], scorer)
        with self.assertRaises(ParseError):
            target_argmax_agreement([], scorer, [])
        with self.assertRaises(ParseError):
            target_argmax_agreement([1], scorer, [])
        with self.assertRaises(ParseError):
            constant_predictor([])

    def test_dale_readout_respects_sign_box_and_grid(self):
        gram, rhs, inhibitory = _diagonal_fit_case()
        weights = fit_dale_readout(gram, rhs, 0.5, inhibitory)
        flat = [w for row in weights for w in row]
        self.assertEqual(flat, [f32(decode_q88(encode_q88(w))) for w in flat])
        self.assertEqual(sign_census(weights, inhibitory)["dale_violations"], 0)
        # E neurons: positive rhs fits positive, negative rhs clamps to 0.
        self.assertGreater(weights[0][0], 0.0)
        self.assertEqual(weights[0][1], 0.0)
        # I neurons: positive rhs clamps to 0.
        self.assertEqual(weights[12][0], 0.0)

    def test_readout_target_matches_distill(self):
        stim = [0.0] * N_NEURONS
        stim[1] = 0.25  # power_w axon
        stim[2] = 0.5  # gpu_temp_c axon
        self.assertEqual(readout_target(stim), (0.625, 0.5, 0.25))
        stim[1] = stim[2] = 1.0
        self.assertEqual(readout_target(stim), (0.0, 1.0, 1.0))

    def test_metrics_weight_sessions_equally(self):
        targets = [(0.0, 0.0, 0.0)] * 6
        preds = [(1.0, 0.0, 0.0)] * 2 + [(0.0, 0.0, 0.0)] * 4
        spans = [Span("a", 0, 2), Span("b", 2, 6)]
        metrics = metrics_from_predictions(preds, targets, spans)
        self.assertEqual(metrics["session_weighted_mae"][0], 0.5)
        self.assertAlmostEqual(metrics["pooled_mae"][0], 2 / 6)
        self.assertAlmostEqual(metrics["primary"], 0.5 / 3)

    def test_session_spans_follow_file_order(self):
        spans = session_spans(_fixture_samples())
        self.assertEqual([s.session for s in spans][:2], ["gpu-000001", "gpu-000002"])
        self.assertEqual(spans[0].stop - spans[0].start, 32)

    def test_activity_definitions(self):
        act = activity([0b1, 0b11, 0b0])
        self.assertEqual(act["spikes_per_tick"], 1.0)
        self.assertEqual(act["patterns"], 3)
        self.assertAlmostEqual(act["cofire"], 1 / 2 ** 0.5)
        self.assertEqual(act["silent_neurons"], list(range(2, N_NEURONS)))
        with self.assertRaises(ParseError):
            activity([])

    def test_promising_verdict(self):
        ref = {"primary": 1.0, "session_weighted_mae": [1.0, 1.0, 1.0]}
        good = {"primary": 0.9, "session_weighted_mae": [0.9, 0.95, 0.85]}
        bad_head = {"primary": 0.9, "session_weighted_mae": [0.7, 1.1, 0.9]}
        self.assertTrue(promising(good, ref)["promising"])
        self.assertFalse(promising(bad_head, ref)["promising"])
        self.assertFalse(promising(ref, ref)["promising"])

    def test_promising_never_passes_a_regression_from_a_zero_reference(self):
        # A head the reference gets exactly right has no finite relative
        # degradation; any candidate error there must fail the per-head limit.
        zero_head_ref = {"primary": 0.2, "session_weighted_mae": [0.0, 0.3, 0.3]}
        regressed = {"primary": 0.1, "session_weighted_mae": [0.05, 0.1, 0.15]}
        verdict = promising(regressed, zero_head_ref)
        self.assertFalse(verdict["promising"])
        self.assertIsNone(verdict["per_head_degradation"][0])
        self.assertIsNone(verdict["worst_head_degradation"])
        self.assertEqual(fmt_ratio(verdict["worst_head_degradation"]),
                         "undefined (reference error is 0)")
        # Zero against zero is a real 0.0, and a zero-error reference cannot be improved on.
        zeros = {"primary": 0.0, "session_weighted_mae": [0.0, 0.0, 0.0]}
        same = promising(zeros, zeros)
        self.assertEqual(same["per_head_degradation"], [0.0, 0.0, 0.0])
        self.assertEqual(same["primary_improvement"], 0.0)
        self.assertFalse(same["promising"])
        self.assertIsNone(promising(regressed, zeros)["primary_improvement"])
        self.assertEqual(fmt_ratio(0.0948), "+9.48%")


class TestCli(unittest.TestCase):
    def test_fixture_run_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "one", Path(tmp) / "two"
            self.assertEqual(cli_main(["--out-dir", str(one)]), 0)
            self.assertEqual(cli_main(["--out-dir", str(two)]), 0)
            self.assertEqual(
                (one / "comparison.json").read_bytes(), (two / "comparison.json").read_bytes()
            )
            doc = json.loads((one / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue((one / "report.md").is_file())
            self.assertTrue((one / "timing.json").is_file())
        self.assertEqual(doc["arm_order"][:2], ["ff-shipped", "ff-refit"])
        self.assertIsInstance(doc["selection"]["verdict_vs_ff_refit"]["promising"], bool)
        self.assertEqual(doc["arms"]["ff-shipped"]["decision_agreement_vs_ff_shipped"], 1.0)
        for name in doc["arm_order"]:
            self.assertEqual(doc["arms"][name]["readout"]["sign_census"]["dale_violations"], 0)

    def test_input_without_test_episodes_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                cli_main(["--jsonl", str(REPLAY_FIXTURE), "--out-dir", tmp]), 2
            )

    def test_emit_checkpoint_refuses_k_and_i_drive_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            for extra in (["--k", "none"], ["--k", "4"], ["--i-drive", "0.05"]):
                self.assertEqual(
                    cli_main([*extra, "--out-dir", str(Path(tmp) / "out"),
                              "--emit-checkpoint", str(bundle)]),
                    2,
                )
            self.assertFalse(bundle.exists())

    def test_protocol_and_input_refuse_empty_gains_and_empty_splits(self):
        bank = load_model_bank(SHIPPED_MANIFEST)
        entry = select_entry(bank, None)
        raw, grid = prepare_base(entry)
        splits, digest = load_splits(FIXTURE_JSONL)
        with self.assertRaises(ParseError):
            Protocol(k=4, i_drive=0.05, gains=())
        with self.assertRaises(ParseError):
            EvalInput(splits={**splits, "val": []}, sha256=digest, jsonl=FIXTURE_JSONL)
        comparison = Comparison(
            AttestedBase(bank=bank, entry=entry, raw=raw, grid=grid),
            EvalInput(splits=splits, sha256=digest, jsonl=FIXTURE_JSONL),
            Protocol(k=4, i_drive=0.05, gains=(0.125,)),
        )
        self.assertEqual(comparison.gains, (0.125,))

    def test_emitted_bundle_attests_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            bundle = Path(tmp) / "bundle"
            self.assertEqual(
                cli_main(["--out-dir", str(out), "--emit-checkpoint", str(bundle)]), 0
            )
            bank = load_model_bank(bundle / "model_bank.json")
            self.assertEqual(bank.ids(), ("merged_v2_rec_v1",))
            for name in UNCHANGED_MEM_IMAGES:
                self.assertEqual(
                    (bundle / name).read_bytes().replace(b"\r\n", b"\n"),
                    (SHIPPED_DIR / name).read_bytes().replace(b"\r\n", b"\n"),
                )
            self.assertEqual(
                replay_main(["--bank-manifest", str(bundle / "model_bank.json"),
                             "--out-dir", str(Path(tmp) / "replay")]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
