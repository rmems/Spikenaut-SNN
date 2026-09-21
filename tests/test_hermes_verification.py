"""Hermes verification regression tests."""

from __future__ import annotations


from pathlib import Path


import socket


import pytest


from tests.hermes_fixture import FakeRuntime


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
    from tools.anticipation.hermes_campaign import _run_verifier

    command = [sys.executable, "-c", f"import sys; sys.{stream}.write('x' * 65537)"]
    with pytest.raises(RuntimeError, match="output exceeded 65536 bytes"):
        _run_verifier(command)
