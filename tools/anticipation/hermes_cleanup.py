"""Bounded cleanup for one Hermes task and its locally owned model."""

from datetime import datetime, timezone
from http.client import HTTPException
import signal
import subprocess
import time


TASK_CLEANUP_ERRORS = (HTTPException, OSError, RuntimeError, TypeError, ValueError)
TASK_CONTROL_FLOW_ERRORS = (KeyboardInterrupt, SystemExit, GeneratorExit)
TASK_FINALIZATION_ERRORS = (
    TASK_CLEANUP_ERRORS + (subprocess.SubprocessError,) + TASK_CONTROL_FLOW_ERRORS
)
TASK_REAP_TIMEOUT_SECONDS = 1.0


def finish_task_cleanup(
    process,
    record,
    origin,
    event,
    pending_error,
    *,
    runtime,
    records,
    signal_process_group,
):
    """Stop descendants, release the model, and preserve error precedence."""
    deadline = origin + event["cleanup_deadline_s"]
    process_error = _stop_task_process_safely(
        process, record, deadline, signal_process_group
    )
    try:
        model_error = _close_task_model_safely(runtime, record, origin, event)
    finally:
        _finalize_task_record(records, record, origin)
    error_to_raise = _cleanup_error_to_raise(pending_error, process_error, model_error)
    if error_to_raise is not None:
        raise error_to_raise
    return _preferred_cleanup_error(pending_error, model_error)


def _stop_task_process_safely(process, record, deadline, signal_process_group):
    try:
        _stop_task_process(process, record, deadline, signal_process_group)
    except TASK_FINALIZATION_ERRORS as error:
        _record_task_cleanup_error(record, "process_cleanup_error", error)
        _retry_task_process_cleanup(process, record, deadline, signal_process_group)
        return error
    return None


def _retry_task_process_cleanup(process, record, deadline, signal_process_group):
    if process is None or process.poll() is not None:
        return
    try:
        _stop_task_process(
            process,
            record,
            deadline,
            signal_process_group,
            reap_timeout=TASK_REAP_TIMEOUT_SECONDS,
        )
    except TASK_FINALIZATION_ERRORS as error:
        _record_task_cleanup_error(record, "process_cleanup_retry_error", error)


def _stop_task_process(
    process, record, deadline, signal_process_group, *, reap_timeout=None
):
    if process is None:
        return
    leader_running = process.poll() is None
    # Descendants can survive after the Hermes group leader exits.
    # Always signal the isolated process group before model cleanup.
    signal_process_group(process, signal.SIGKILL)
    if leader_running:
        timeout = (
            max(0, deadline - time.monotonic())
            if reap_timeout is None
            else reap_timeout
        )
        process.wait(timeout)
        record["forced_kill"] = True


def _close_task_model_safely(runtime, record, origin, event):
    try:
        _close_task_model(runtime, record, origin, event)
    except TASK_FINALIZATION_ERRORS as error:
        _record_task_cleanup_error(record, "model_cleanup_error", error)
        return error
    return None


def _close_task_model(runtime, record, origin, event):
    record["model_cleanup"] = runtime.close(
        deadline=origin + event["cleanup_deadline_s"]
    )
    record["model_cleanup_end_s"] = time.monotonic() - origin
    if record["model_cleanup_end_s"] > event["cleanup_deadline_s"]:
        raise RuntimeError("owned model cleanup exceeded the 130s session deadline")


def _record_task_cleanup_error(record, key, error):
    record[key] = f"{type(error).__name__}: {error}"
    record["status"] = "invalid"
    record["workload_status"] = "invalid"


def _cleanup_error_to_raise(pending_error, process_error, model_error):
    for error in (process_error, model_error):
        if isinstance(error, TASK_CONTROL_FLOW_ERRORS):
            return error
    if pending_error is not None:
        return None
    if process_error is not None:
        return process_error
    if model_error is not None and not isinstance(model_error, TASK_CLEANUP_ERRORS):
        return model_error
    return None


def _preferred_cleanup_error(*errors):
    return next((error for error in errors if error is not None), None)


def _finalize_task_record(records, record, origin):
    record.setdefault("ended_at_utc", datetime.now(timezone.utc).isoformat())
    record.setdefault("actual_end_s", time.monotonic() - origin)
    if not records or records[-1] is not record:
        records.append(record)
