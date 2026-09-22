"""Hermes verification regression tests."""

from __future__ import annotations


from pathlib import Path


import socket


import pytest


from tests.hermes_fixture import FakeRuntime


def test_verification_does_not_swallow_keyboard_interrupt(monkeypatch):
    from tools.anticipation import task_verification

    def interrupt_verifier(_plan):
        raise KeyboardInterrupt

    monkeypatch.setattr(task_verification, "_verify_json_output", interrupt_verifier)

    with pytest.raises(KeyboardInterrupt):
        task_verification.verify_fixture({}, {"kind": "json-output"})


def test_runtime_roots_include_copied_virtual_environment(tmp_path, monkeypatch):
    from tools.anticipation import task_verification

    base = tmp_path / "base"
    prefix = tmp_path / "venv"
    interpreter = prefix / "bin" / "python"
    base.mkdir()
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"copied interpreter")
    monkeypatch.setattr(task_verification.sys, "base_prefix", str(base))
    monkeypatch.setattr(task_verification.sys, "prefix", str(prefix))
    monkeypatch.setattr(task_verification.sys, "executable", str(interpreter))

    roots = task_verification._verification_runtime_roots()

    assert any(interpreter.is_relative_to(root) for root in roots)


def test_json_verifier_rejects_boolean_in_place_of_number(tmp_path):
    from tools.anticipation.task_verification import verify_fixture

    output = tmp_path / "output.json"
    output.write_text('[{"id":"item-00000","score":false}]\n')
    result = verify_fixture(
        {},
        {
            "kind": "json-output",
            "path": str(output),
            "expected": [{"id": "item-00000", "score": 0}],
        },
    )

    assert result == {"status": "failed", "reason": "output mismatch"}


def test_python_verifier_requires_completion_sentinel(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    verifier = Path(stimulus._verifier_plan["path"])
    scratch = Path(session["scratch_path"])
    assert not verifier.is_relative_to(scratch)
    assert not any(
        "['beta', 'alpha', 'gamma']" in path.read_text() for path in scratch.iterdir()
    )
    (Path(session["scratch_path"]) / "transform.py").write_text("raise SystemExit(0)\n")

    verification = stimulus._verify_fixture(session)

    assert verification["status"] == "failed"
    assert verification["exit_code"] != 0
    assert verification["stdout"] == ""


def test_python_candidate_cannot_exit_the_parent_assertion(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    (Path(session["scratch_path"]) / "transform.py").write_text(
        'import os\nprint(\'["beta","alpha","gamma"]\', flush=True)\nos._exit(0)\n'
    )

    verification = stimulus._verify_fixture(session)

    assert verification["status"] == "failed"
    assert verification["exit_code"] != 0
    assert verification["stdout"] == ""


def test_python_verifier_sandbox_blocks_host_writes_and_network(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    outside = tmp_path / "outside.txt"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    port = listener.getsockname()[1]
    candidate = Path(session["scratch_path"]) / "transform.py"
    candidate.write_text(
        "from pathlib import Path\n"
        "import socket\n"
        "def normalize(words):\n"
        "    try:\n"
        f"        Path({str(outside)!r}).write_text('escaped')\n"
        "    except OSError:\n"
        "        pass\n"
        "    try:\n"
        f"        with socket.create_connection(('127.0.0.1', {port}), timeout=.05) as client:\n"
        "            client.sendall(b'escaped')\n"
        "    except OSError:\n"
        "        pass\n"
        "    return [word.strip().lower() for word in words if word.strip()]\n"
    )

    _install_trusted_sandbox_probe(
        stimulus, Path(session["scratch_path"], "transform.py").read_text()
    )
    try:
        verification = stimulus._verify_fixture(session)
        with pytest.raises(TimeoutError):
            listener.accept()
    finally:
        listener.close()

    assert verification["status"] == "passed"
    assert outside.exists() is False


def test_python_verifier_sandbox_applies_resource_limits(tmp_path):
    from tools.anticipation.hermes_campaign import (
        HermesStimulus,
        VERIFIER_AS_LIMIT_BYTES,
        VERIFIER_CPU_LIMIT_SECONDS,
        VERIFIER_FSIZE_LIMIT_BYTES,
        VERIFIER_NPROC_LIMIT,
        build_hermes_campaign,
    )

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    candidate = Path(session["scratch_path"]) / "transform.py"
    candidate.write_text(
        "import resource\n"
        "def normalize(words):\n"
        f"    assert resource.getrlimit(resource.RLIMIT_AS)[0] == {VERIFIER_AS_LIMIT_BYTES}\n"
        f"    assert resource.getrlimit(resource.RLIMIT_NPROC)[0] == {VERIFIER_NPROC_LIMIT}\n"
        f"    assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == {VERIFIER_FSIZE_LIMIT_BYTES}\n"
        f"    assert resource.getrlimit(resource.RLIMIT_CPU)[0] == {VERIFIER_CPU_LIMIT_SECONDS}\n"
        "    return [word.strip().lower() for word in words if word.strip()]\n"
    )

    _install_trusted_sandbox_probe(
        stimulus, Path(session["scratch_path"], "transform.py").read_text()
    )
    verification = stimulus._verify_fixture(session)

    assert verification["status"] == "passed"


def test_python_verifier_denies_candidate_fork(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        item
        for item in build_hermes_campaign(tmp_path)["sessions"]
        if item["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    (Path(session["scratch_path"]) / "transform.py").write_text(
        "import os\n"
        "def normalize(words):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except PermissionError:\n"
        "        return [word.strip().lower() for word in words if word.strip()]\n"
        "    if pid == 0:\n"
        "        os._exit(0)\n"
        "    os.waitpid(pid, 0)\n"
        "    raise AssertionError('candidate fork was allowed')\n"
    )
    _install_trusted_sandbox_probe(
        stimulus, Path(session["scratch_path"], "transform.py").read_text()
    )
    assert stimulus._verify_fixture(session)["status"] == "passed"


def test_python_candidate_cannot_forge_frame_nonce(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        s
        for s in build_hermes_campaign(tmp_path)["sessions"]
        if s["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    Path(session["scratch_path"], "transform.py").write_text(
        "import sys,json,os\n"
        "def normalize(words):\n"
        "    nonce = sys._getframe(1).f_locals['nonce']\n"
        "    print(json.dumps({'nonce':nonce,'result':['beta','alpha','gamma']}), flush=True)\n"
        "    os._exit(0)\n"
    )
    assert stimulus._verify_fixture(session)["status"] == "failed"


def test_python_verifier_caps_outer_output(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        s
        for s in build_hermes_campaign(tmp_path)["sessions"]
        if s["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    Path(session["scratch_path"], "transform.py").write_text(
        "import os\n"
        "def normalize(words):\n"
        "    with open('/proc/' + str(os.getppid()) + '/fd/1', 'w') as out:\n"
        "        out.write('x' * (2 * 1024 * 1024))\n"
        "    return []\n"
    )
    result = stimulus._verify_fixture(session)
    assert result["status"] == "failed"
    assert len(result.get("stdout", "")) <= 65536


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_verifier_outer_capture_rejects_oversized_stream(stream):
    import sys
    from tools.anticipation.task_verification import _run_verifier

    command = [sys.executable, "-c", f"import sys; sys.{stream}.write('x' * 65537)"]
    with pytest.raises(RuntimeError, match="output exceeded 65536 bytes"):
        _run_verifier(command)


def test_python_candidate_cannot_forge_all_worker_results(tmp_path):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        s
        for s in build_hermes_campaign(tmp_path)["sessions"]
        if s["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    Path(session["scratch_path"], "transform.py").write_text(
        "import sys,json,os\n"
        "def normalize(words):\n"
        "    frame = sys._getframe()\n"
        "    while 'inputs' not in frame.f_locals:\n"
        "        frame = frame.f_back\n"
        "    values = frame.f_locals['inputs']\n"
        "    print(json.dumps([[w.strip().lower() for w in row if w.strip()] for row in values]), flush=True)\n"
        "    os._exit(0)\n"
    )
    assert stimulus._verify_fixture(session)["status"] == "failed"


def _install_trusted_sandbox_probe(stimulus, probe):
    """Only test-owned code replaces the trusted runner, never an agent candidate."""
    import hashlib
    from tools.anticipation import verifier_runner

    runner = Path(stimulus._verifier_plan["path"])
    trusted = (
        Path(verifier_runner.__file__)
        .read_text()
        .split('if __name__ == "__main__":')[0]
    )
    runner.write_text(
        trusted
        + "\nrestrict_candidate()\n"
        + probe
        + "\nprint(json.dumps(normalize([' Beta ', '', 'ALPHA', ' gamma ']), separators=(',', ':')))\n"
    )
    stimulus._verifier_plan["sha256"] = hashlib.sha256(runner.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "source",
    [
        "def normalize(words):\n    return [w.strip().lower() for w in words if w.strip()]\n",
        "def normalize(words):\n    result = []\n    for word in words:\n        word = word.strip().lower()\n        if word:\n            result.append(word)\n    return result\n",
    ],
)
def test_python_verifier_accepts_pure_normalization(tmp_path, source):
    assert _verify_candidate_source(tmp_path, source)["status"] == "passed"


@pytest.mark.parametrize(
    "statement",
    [
        "hidden = __builtins__",
        "hidden = normalize.__globals__",
        "hidden = ().__class__.__base__.__subclasses__()",
        "hidden = globals()",
        "print('forged result')",
        "import sys",
    ],
)
def test_python_verifier_rejects_worker_access(tmp_path, statement):
    source = (
        "def normalize(words):\n    "
        + statement
        + "\n    return [w.strip().lower() for w in words if w.strip()]\n"
    )
    assert _verify_candidate_source(tmp_path, source)["status"] == "failed"


def _verify_candidate_source(tmp_path, source):
    from tools.anticipation.hermes_campaign import HermesStimulus, build_hermes_campaign

    session = next(
        s
        for s in build_hermes_campaign(tmp_path)["sessions"]
        if s["task"]["family"] == "python-bugfix"
    )
    stimulus = HermesStimulus(
        tmp_path, hermes_executable=tmp_path / "hermes", runtime=FakeRuntime()
    )
    stimulus.seed(session["seed"])
    stimulus.prepare_session(session)
    Path(session["scratch_path"], "transform.py").write_text(source)
    return stimulus._verify_fixture(session)
