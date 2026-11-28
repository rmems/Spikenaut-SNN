import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from tools.temporal_replay import MAX_TICK, main


def sample(tick, raw):
    return {"tick": tick, "raw": raw, "stim": [0, 0, 1.5, 0, 0], "gain": 2.5}


class TemporalReplayTest(unittest.TestCase):
    def convert(self, rows, encoding, fmt="replay", separator="\n"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(separator.join(json.dumps(r, ensure_ascii=False) for r in rows),
                            encoding="utf-8", newline="")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main([str(path), "--encoding", encoding, "--format", fmt])
            return code, out.getvalue(), err.getvalue()

    def test_ties_are_deferred_and_silence_keeps_its_frame(self):
        low, high = [0, 8.527, 0, 180, 405], [75, 302.845, 0, 180, 405]
        rows = [sample(i, raw) for i, raw in enumerate([low, high, high, high], 17)]
        code, out, err = self.convert(rows, "delta")
        self.assertEqual(code, 0, err)
        self.assertIn('"stim":[', out)
        self.assertEqual([json.loads(row)["stim"] for row in out.splitlines()],
                         [[0]*5, [1,0,0,0,0], [0,1,0,0,0], [0]*5])
        self.assertEqual(self.convert(rows, "delta", "events")[1], "18\t0\n19\t1\n")
        self.assertEqual(self.convert(rows, "logged", "events")[1], "17\t2\n18\t2\n19\t2\n20\t2\n")

    def test_malformed_or_discontinuous_capture_never_emits_a_prefix(self):
        first = sample(1, [1, 20, 30, 400, 500])
        second = {**first, "tick": 2}
        for bad in ({"tick": 2, "event": "poll_fail"}, {**second, "tick": 4},
                    {**second, "raw": None}, {**second, "raw": [True]*5},
                    {**second, "stim": [1, 2, 0, 0, 0]}, {**second, "stim": [0.1, 0, 0, 0, 0]},
                    {**second, "raw": [float("nan")]*5}):
            with self.subTest(bad=bad):
                code, out, err = self.convert([first, bad], "delta")
                self.assertEqual((code, out), (2, ""))
                self.assertIn("line 2", err)

    def test_only_lf_or_crlf_delimits_capture_rows(self):
        first = sample(1, [1, 20, 30, 400, 500])
        first["note"] = "literal\u2028separator inside JSON string"
        second = {**first, "tick": 2}
        for separator in ("\n", "\r\n"):
            code, out, err = self.convert([first, second], "logged", separator=separator)
            self.assertEqual(code, 0, err)
            self.assertEqual(len(out.splitlines()), 2)
        code, out, err = self.convert([first, second], "logged", separator="\v")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("line 1", err)

    def test_tick_and_exclusive_end_fit_julia_int64(self):
        raw = [1, 20, 30, 400, 500]
        self.assertEqual(self.convert([sample(MAX_TICK, raw)], "logged")[0], 0)
        code, out, err = self.convert([sample(MAX_TICK + 1, raw)], "logged")
        self.assertEqual((code, out), (2, ""))
        self.assertIn(str(MAX_TICK), err)
