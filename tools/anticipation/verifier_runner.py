"""Trusted functional verifier; candidate workers receive inputs, never expected results."""

import ast
import ctypes
import json
from pathlib import Path
import secrets
import subprocess
import sys

READY = "candidate-validation-complete"


def restrict_candidate():
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("cannot initialize candidate seccomp filter")
    try:
        _deny_syscalls(lib, context)
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("cannot load candidate seccomp filter")
    finally:
        lib.seccomp_release(context)


def _deny_syscalls(lib, context):
    names = (
        "fork",
        "vfork",
        "clone",
        "clone3",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "pidfd_getfd",
        "kill",
        "tkill",
        "tgkill",
    )
    for name in names:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and lib.seccomp_rule_add(context, 0x00050001, number, 0) != 0:
            raise RuntimeError("cannot restrict candidate system calls")


def validate_function(function):
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    assert not function.decorator_list and not function.args.defaults
    assert all(default is None for default in function.args.kw_defaults)
    assert function.returns is None
    assert all(argument.annotation is None for argument in arguments)


def load_candidate(candidate):
    tree = ast.parse(candidate.read_text(), filename=str(candidate))
    allowed = (ast.FunctionDef, ast.Import, ast.ImportFrom)
    assert tree.body and all(isinstance(node, allowed) for node in tree.body)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert sum(node.name == "normalize" for node in functions) == 1
    for function in functions:
        validate_function(function)
    namespace = {}
    exec(compile(tree, str(candidate), "exec"), namespace)
    return namespace["normalize"]


def candidate_worker(candidate):
    restrict_candidate()
    normalize = load_candidate(candidate)
    print(READY, flush=True)
    inputs = json.loads(sys.stdin.readline())
    # The worker and everything it emits are untrusted. No expected output or
    # attestation secret is present in this interpreter.
    results = [normalize(words) for words in inputs]
    print(json.dumps(results, separators=(",", ":")))


def test_inputs():
    values = [" Beta ", "", "ALPHA", " gamma "]
    randomized = [" " + secrets.token_hex(12).upper() + " " for _ in range(8)]
    return [
        values,
        ["", " \t ", *reversed(randomized), ""],
        [],
        ["Repeated", "Repeated"],
    ]


def verify(candidate):
    inputs = test_inputs()
    expected = [
        [word.strip().lower() for word in words if word.strip()] for words in inputs
    ]
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", __file__, "--child", str(candidate)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    ready = child.stdout.readline()
    if ready != READY + "\n":
        child.kill()
        child.communicate()
        raise SystemExit("candidate did not complete validation")
    stdout, stderr = child.communicate(json.dumps(inputs) + "\n")
    if child.returncode != 0 or stderr:
        raise SystemExit("candidate evaluation failed")
    if json.loads(stdout) != expected:
        raise SystemExit("candidate output mismatch")
    print(json.dumps(expected[0], separators=(",", ":")))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        candidate_worker(Path(sys.argv[2]))
    else:
        verify(Path(sys.argv[1]))
