"""Trusted functional verifier; candidate workers receive inputs, never expected results."""

import ast
import ctypes
import json
from pathlib import Path
import secrets
import subprocess
import sys
import types

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


SAFE_BUILTINS = {
    "list": list,
    "tuple": tuple,
    "str": str,
    "len": len,
    "bool": bool,
    "sorted": sorted,
    "reversed": reversed,
}
SAFE_METHODS = {"strip", "lower", "upper", "append"}
SAFE_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.For,
    ast.If,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.Attribute,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.NotIn,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.IfExp,
}


def validate_expression(node):
    if type(node) not in SAFE_NODES:
        raise ValueError(f"unsupported candidate syntax: {type(node).__name__}")
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ValueError("private names are not available to candidates")
    if isinstance(node, ast.Attribute):
        if node.attr not in SAFE_METHODS or not isinstance(node.ctx, ast.Load):
            raise ValueError("only normalization methods are available")
    if isinstance(node, ast.Call):
        validate_call(node)


def validate_call(node):
    if isinstance(node.func, ast.Name):
        if node.func.id not in SAFE_BUILTINS:
            raise ValueError("candidate called an unsupported function")
    elif not isinstance(node.func, ast.Attribute):
        raise ValueError("candidate called an unsupported expression")
    if node.keywords:
        raise ValueError("candidate keyword calls are unsupported")


def load_candidate(candidate):
    source = candidate.read_bytes()
    if len(source) > 65536:
        raise ValueError("candidate source exceeds 64 KiB")
    tree = ast.parse(source, filename=str(candidate))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("candidate must define only the pure normalize function")
    function = tree.body[0]
    if function.name != "normalize":
        raise ValueError("candidate must define normalize")
    validate_function(function)
    validate_candidate_tree(tree, function)
    module_code = compile(tree, str(candidate), "exec")
    func_code = next(
        (
            const
            for const in module_code.co_consts
            if isinstance(const, types.CodeType) and const.co_name == "normalize"
        ),
        None,
    )
    if func_code is None:
        raise ValueError("candidate must define normalize")
    return types.FunctionType(func_code, {"__builtins__": dict(SAFE_BUILTINS)})


def validate_candidate_tree(tree, function):
    nodes = list(ast.walk(tree))
    if len(nodes) > 2048:
        raise ValueError("candidate syntax exceeds the node budget")
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("nested candidate functions are unsupported")
        validate_expression(node)


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


def verify():
    inputs = test_inputs()
    expected = [
        [word.strip().lower() for word in words if word.strip()] for words in inputs
    ]
    child = subprocess.Popen(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        [sys.executable, "-I", "-B", "/harness/runner.py", "--child"],
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
    candidate = Path("/work/transform.py")
    if sys.argv[1:] == ["--child"]:
        candidate_worker(candidate)
    elif not sys.argv[1:]:
        verify()
    else:
        raise SystemExit("unsupported verifier arguments")
