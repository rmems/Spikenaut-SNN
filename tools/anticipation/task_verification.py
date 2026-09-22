"""Fixture generation and isolated, bounded candidate verification."""

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

VERIFIER_AS_LIMIT_BYTES = 512 * 1024 * 1024
VERIFIER_NPROC_LIMIT = 8192
VERIFIER_FSIZE_LIMIT_BYTES = 1024 * 1024
VERIFIER_CPU_LIMIT_SECONDS = 4


def write_fixture(session, scratch):
    family = session["task"]["family"]
    count = session["task"]["fixture_records"]
    # NOSONAR python:S2245 -- deterministic non-security test-fixture RNG; the
    # seed is the declared campaign seed and the output is experiment data
    # (fixture rows/records/order), never a secret, token, nonce, salt, or
    # authorization value. A CSPRNG would break the seed->fixture replay
    # contract the campaign and tests depend on.
    rng = random.Random(session["seed"])  # NOSONAR
    if family == "csv-aggregation":
        rows = ["category,amount"]
        totals = {}
        for _ in range(count):
            category = rng.choice(("alpha", "beta", "delta", "gamma"))  # NOSONAR
            amount = rng.randint(1, 99)  # NOSONAR
            rows.append(f"{category},{amount}")
            totals[category] = totals.get(category, 0) + amount
        (scratch / "input.csv").write_text("\n".join(rows) + "\n")
        expected = dict(sorted(totals.items()))
    elif family == "json-transformation":
        records = [
            {
                "id": f"item-{i:05}",
                "score": rng.randint(0, 1000),  # NOSONAR
                "enabled": rng.choice((True, False)),  # NOSONAR
            }
            for i in range(count)
        ]
        rng.shuffle(records)  # NOSONAR python:S2245 -- deterministic fixture RNG
        (scratch / "input.json").write_text(json.dumps(records, indent=2) + "\n")
        expected = [
            {"id": r["id"], "score": r["score"]}
            for r in sorted(records, key=lambda x: x["id"])
            if r["enabled"]
        ]
    else:
        return _python_fixture(session, scratch, count)
    return {
        "kind": "json-output",
        "path": str(scratch / "output.json"),
        "expected": expected,
    }


def verify_fixture(session, plan):
    if not isinstance(plan, dict):
        return {"status": "failed", "reason": "missing verifier plan"}
    try:
        if plan["kind"] == "json-output":
            return _verify_json_output(plan)
        verifier = Path(plan["path"])
        if hashlib.sha256(verifier.read_bytes()).hexdigest() != plan["sha256"]:
            return {"status": "failed", "reason": "fixed verifier was modified"}
        command = verification_command(session, verifier)
        completed = _run_verifier(command)
        expected_stdout = json.dumps(plan["expected"], separators=(",", ":")) + "\n"
        verified = completed.returncode == 0 and completed.stderr == ""
        verified = verified and completed.stdout == expected_stdout
        return {
            "status": "passed" if verified else "failed",
            "kind": "fixed-python-test",
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (
        LookupError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
        }


def verification_command(session, verifier):
    sandbox, limiter = _verification_tools()
    runtime_roots = _verification_runtime_roots()
    # NOSONAR python:S5443 -- "/tmp" here is a private tmpfs mounted inside a
    # fresh --unshare-all bubblewrap mount namespace, not the shared, world-
    # writable host /tmp. No other process can observe or write this mount, so
    # there is no shared-temp-directory risk. HOME (below) points at this same
    # private mount.
    command = [
        limiter,
        f"--as={VERIFIER_AS_LIMIT_BYTES}",
        f"--nproc={VERIFIER_NPROC_LIMIT}",
        f"--fsize={VERIFIER_FSIZE_LIMIT_BYTES}",
        f"--cpu={VERIFIER_CPU_LIMIT_SECONDS}",
        "--",
        sandbox,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",  # NOSONAR
    ]
    for root in runtime_roots:
        command.extend(("--ro-bind", str(root), str(root)))
    loader_cache = Path("/etc/ld.so.cache")
    if loader_cache.exists():
        command.extend(("--ro-bind", str(loader_cache), str(loader_cache)))
    command.extend(
        (
            "--ro-bind",
            str(verifier),
            "/harness/runner.py",
            "--ro-bind",
            str(Path(session["scratch_path"]) / "transform.py"),
            "/work/transform.py",
            "--chdir",
            "/work",
            "--setenv",
            "HOME",
            "/tmp",  # NOSONAR python:S5443 -- private in-namespace tmpfs; see above
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            "/harness/runner.py",
        )
    )
    return command


def _verification_runtime_roots():
    runtime_roots = []
    python_base = Path(sys.base_prefix)
    python_prefix = Path(sys.prefix)
    interpreter = Path(sys.executable).resolve()
    candidates = [
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        python_base,
        python_base.resolve(),
        python_prefix,
        python_prefix.resolve(),
        interpreter.parent,
    ]
    candidates.extend(_python_loader_roots(python_base))
    for root in candidates:
        if not root.exists() or any(
            root.is_relative_to(bound) for bound in runtime_roots
        ):
            continue
        runtime_roots = [
            bound for bound in runtime_roots if not bound.is_relative_to(root)
        ]
        runtime_roots.append(root)
    return runtime_roots


def _python_loader_roots(python_base):
    for prefix in python_base.parents:
        loader_root = prefix / "lib"
        if (loader_root / "ld.so").exists():
            return [loader_root]
    return []


def _verify_json_output(plan):
    actual = json.loads(Path(plan["path"]).read_text())
    if not _same_json_value(actual, plan["expected"]):
        return {"status": "failed", "reason": "output mismatch"}
    if isinstance(plan["expected"], dict) and list(actual) != list(plan["expected"]):
        return {"status": "failed", "reason": "output key order mismatch"}
    return {"status": "passed", "kind": "direct-json-comparison"}


def _same_json_value(actual, expected):
    if isinstance(expected, dict):
        return _same_json_object(actual, expected)
    if isinstance(expected, list):
        return _same_json_array(actual, expected)
    if _is_json_number(expected):
        return _is_json_number(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _same_json_object(actual, expected):
    return (
        isinstance(actual, dict)
        and set(actual) == set(expected)
        and all(_same_json_value(actual[key], value) for key, value in expected.items())
    )


def _same_json_array(actual, expected):
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            _same_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    )


def _is_json_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _run_verifier(command):
    # Regular files are subject to the sandbox RLIMIT_FSIZE. Never accumulate
    # candidate-controlled outer stdout/stderr in an unbounded host pipe.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, timeout=5, check=False
        )
        captured = []
        for stream in (stdout, stderr):
            stream.seek(0)
            value = stream.read(65537)
            if len(value) > 65536:
                raise RuntimeError("verifier output exceeded 65536 bytes")
            captured.append(value.decode("utf-8", errors="replace"))
    completed.stdout, completed.stderr = captured
    return completed


def _python_fixture(session, scratch, count):
    lines = [
        "Specification: normalize each input word by stripping surrounding whitespace, "
        "convert it to lowercase, discard empty values, and preserve input order."
    ]
    lines.extend(
        f"Example note {i:04}: normalization is deterministic and must not sort values."
        for i in range(1, count)
    )
    (scratch / "specification.txt").write_text("\n".join(lines) + "\n")
    (scratch / "transform.py").write_text(
        "def normalize(words):\n"
        "    return sorted(w.strip().upper() for w in words if w.strip())\n"
    )
    home = Path(session["hermes_home"]).resolve()
    verifier = (home / "harness-runner.py").resolve()
    if not verifier.is_relative_to(home):
        raise ValueError("verifier path escapes hermes_home")
    verifier.write_bytes(Path(__file__).with_name("verifier_runner.py").read_bytes())
    return {
        "kind": "fixed-python-test",
        "candidate_contract": "pure-normalization-v1",
        "path": str(verifier),
        "sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
        "expected": ["beta", "alpha", "gamma"],
    }


def _verification_tools():
    sandbox = shutil.which("bwrap")
    if sandbox is None:
        raise RuntimeError("bubblewrap is required for Python verification")
    limiter = shutil.which("prlimit")
    if limiter is None:
        raise RuntimeError("prlimit is required for Python verification")
    return sandbox, limiter
