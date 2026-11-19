"""A regression: the terminal-worker reaper must signal the BOUND
worker (not the launcher) and must never fire a breach off a dead launcher.

The defect shape: a launcher spawns a worker, the worker binds itself, the run
completes, and the launcher dies — the old reaper SIGKILLed the dead launcher,
reported terminated=True, fired a false breach, and left the bound worker alive
holding the board DB.

The predicates (_pid_alive/_pid_recycled/_worker_alive) are faked per scenario;
the REAL _terminate_reclaimed_worker runs so the signal path is exercised.
Scenario kills are injected as ``signal_fn`` — NOT by patching ``_kill_fn``,
which would bypass the pass-through and the real termination path entirely.
"""

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as m

LAUNCHER_PID = 83585
BOUND_PID = 83608
LAUNCHER_FP = "|179441290706"
BOUND_FP = "|179441290801"
ENDED_AT = 1794412999


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _seed_completed_run(
    conn, monkeypatch, *, launcher_alive, bound_alive, bind_event=True, patch_finishing=True,
):
    tid = kb.create_task(conn, title="reaper regression", assignee="worker")
    run_id = 900569
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (id, task_id, profile, status, worker_pid, "
            "worker_started_at, claim_lock, ended_at, started_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, tid, "developer", "done", LAUNCHER_PID, LAUNCHER_FP,
             f"{m._kb._host_prefix()}:{LAUNCHER_PID}", ENDED_AT, 1794412907),
        )
        if bind_event:
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'worker_bound', ?, 1794412908)",
                (tid, run_id,
                 f'{{"pid": {BOUND_PID}, "claim_lock": "x:{BOUND_PID}", '
                 f'"started_at": "{BOUND_FP}"}}'),
            )

    pids = {LAUNCHER_PID: launcher_alive, BOUND_PID: bound_alive}

    monkeypatch.setattr(m._kb, "_pid_alive", lambda pid: pids.get(pid, False))
    monkeypatch.setattr(m, "_pid_recycled", lambda pid, started_at: False)
    monkeypatch.setattr(m, "_worker_alive", lambda pid, started_at: pids.get(pid, False))
    monkeypatch.setattr(m, "_poll_worker_exit", lambda pid, started_at: not pids.get(pid, True))
    if patch_finishing:
        monkeypatch.setattr(m, "_worker_still_finishing", lambda conn_, row: False)
    return conn, tid, run_id, pids


def _signal_fn(pids, kills, raise_on=None):
    """An ``os.kill`` stand-in passed as ``signal_fn``: records every attempt,
    flips the pid dead, or raises (ProcessLookupError) for raced-exit shapes."""

    def _kill(pid, sig):
        kills.append((pid, sig))
        if raise_on and pid in raise_on:
            raise raise_on[pid]
        pids[pid] = False

    return _kill


def _events(conn, tid):
    return [dict(e) for e in conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id", (tid,))]


def test_dead_launcher_live_bound_reaps_bound_zero_breach(board, monkeypatch):
    conn, tid, run_id, pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=False, bound_alive=True)
    kills = []

    m.reap_terminal_workers(conn, signal_fn=_signal_fn(pids, kills))

    assert (LAUNCHER_PID in kills) is False, "the dead launcher must never be signalled"
    assert [pid for pid, _ in kills] == [BOUND_PID], \
        "the LIVE bound worker must be the reap target"
    events = _events(conn, tid)
    assert not [e for e in events if e["kind"] == "terminal_worker_breach"], \
        "zero breach events: the actually-signalled worker was terminated"
    reaps = [e for e in events if e["kind"] == "terminal_worker_reaped"]
    assert reaps, "the reap must be recorded"
    assert "83608" in (reaps[-1]["payload"] or ""), \
        "the reap event must name the actually-signalled bound pid"
    row = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] is None, "evidence cleared after the reap"


def test_dead_launcher_dead_bound_no_breach(board, monkeypatch):
    conn, tid, run_id, pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=False, bound_alive=False)
    kills = []

    m.reap_terminal_workers(conn, signal_fn=_signal_fn(pids, kills))

    assert kills == [], "nothing live: nothing to signal"
    events = _events(conn, tid)
    assert not [e for e in events if e["kind"] == "terminal_worker_breach"]
    row = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] is None


def test_bound_fingerprint_is_the_bound_process_start(board):
    """The fingerprint paired with the bound pid must be the bound event's own
    started_at — pairing it with the launcher's fingerprint reads a live worker
    as recycled, and the orphan survives silently (the review bug)."""
    conn = board
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES ('t_probe', 900569, 'worker_bound', "
        f"'{{\"pid\": {BOUND_PID}, \"started_at\": \"{BOUND_FP}\"}}', 0)"
    )
    conn.execute(
        "INSERT INTO task_runs (id, task_id, profile, status, worker_pid, "
        "worker_started_at, claim_lock, ended_at, started_at) VALUES "
        "(900569, 't_probe', 'developer', 'done', ?, ?, 'x:1', 1794412999, 1794412907)",
        (LAUNCHER_PID, LAUNCHER_FP),
    )
    run_row = conn.execute("SELECT * FROM task_runs WHERE id=900569").fetchone()
    target, fp = m._bound_worker_target(conn, run_row, LAUNCHER_PID, LAUNCHER_FP)
    assert target == BOUND_PID
    assert fp == BOUND_FP, "the bound worker's own fingerprint must pair with its pid"


def test_bind_event_without_started_at_never_inherits_launcher_fp(board):
    """A legacy bind event with no started_at must not pair the bound pid with
    the LAUNCHER fingerprint: that reads the live worker as recycled and the
    orphan survives silently. The fallback probes the live process instead."""
    conn = board
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES ('t_probe', 900570, 'worker_bound', "
        f"'{{\"pid\": {BOUND_PID}, \"claim_lock\": \"x:{BOUND_PID}\"}}', 0)"
    )
    conn.execute(
        "INSERT INTO task_runs (id, task_id, profile, status, worker_pid, "
        "worker_started_at, claim_lock, ended_at, started_at) VALUES "
        "(900570, 't_probe', 'developer', 'done', ?, ?, 'x:1', 1794412999, 1794412907)",
        (LAUNCHER_PID, LAUNCHER_FP),
    )
    run_row = conn.execute("SELECT * FROM task_runs WHERE id=900570").fetchone()
    target, fp = m._bound_worker_target(conn, run_row, LAUNCHER_PID, LAUNCHER_FP)
    assert target == BOUND_PID
    assert fp != LAUNCHER_FP, "the launcher fingerprint must never pair with the bound pid"


def test_dead_bound_live_launcher_signals_recorded_and_breaches(board, monkeypatch):
    """Mirrored defect shape: bound worker DEAD but
    LAUNCHER alive. The old OR-fold let the launcher probe arm a SIGTERM at the
    dead bound pid and ProcessLookupError reported it "terminated". Now: the
    dead bound pid is never signalled; the actually-alive RECORDED pid is, and
    a delivered termination of the recorded worker_pid IS breach-worthy."""
    conn, tid, run_id, pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=True, bound_alive=False)
    kills = []

    m.reap_terminal_workers(conn, signal_fn=_signal_fn(pids, kills))

    assert (BOUND_PID in kills) is False, "a dead bound pid must never be signalled"
    assert [pid for pid, _ in kills] == [LAUNCHER_PID], \
        "the actually-alive recorded worker_pid is the reap target"
    events = _events(conn, tid)
    breaches = [e for e in events if e["kind"] == "terminal_worker_breach"]
    assert len(breaches) == 1, \
        "a delivered signal to the recorded worker_pid outliving its run is a breach"
    assert f'"pid": {LAUNCHER_PID}' in (breaches[0]["payload"] or ""), \
        "the breach must name the actually-signalled recorded pid"
    row = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] is None


def test_noop_kill_on_dead_pid_never_breaches(board, monkeypatch):
    """Probe says alive but the kill hits ProcessLookupError (raced exit): the
    reap is recorded (the worker IS gone) but signal_delivered=False means the
    no-op kill is NOT evidence of a breach — spec point 3 verbatim."""
    conn, tid, _run_id, pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=True, bound_alive=False, bind_event=False)
    kills = []
    signal_fn = _signal_fn(pids, kills, raise_on={LAUNCHER_PID: ProcessLookupError(LAUNCHER_PID)})

    m.reap_terminal_workers(conn, signal_fn=signal_fn)

    assert [pid for pid, _ in kills] == [LAUNCHER_PID], \
        "exactly one signal attempt, on the recorded pid"
    events = _events(conn, tid)
    assert not [e for e in events if e["kind"] == "terminal_worker_breach"], \
        "a no-op kill on a dead pid is not breach evidence"
    reaps = [e for e in events if e["kind"] == "terminal_worker_reaped"]
    assert reaps, "the (non-)kill is still recorded for the audit trail"
    payload = reaps[-1]["payload"] or ""
    assert "already_gone" in payload and "signal_delivered" in payload, \
        "the reap payload must carry the no-op markers"


def test_finishing_guard_defers_only_on_post_close_heartbeat(board, monkeypatch):
    """The finishing guard must be bounded by post-close liveness. A heartbeat
    AFTER ended_at defers the reap (worker may still be finalising, );
    with no post-close heartbeat the reap proceeds even though worker_result
    never arrived — a run closed without the event is hung, which is what this
    sweep exists to end (#111791, )."""
    conn, tid, run_id, pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=True, bound_alive=False, patch_finishing=False)

    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'heartbeat', NULL, ?)",
            (tid, run_id, ENDED_AT + 10),
        )

    kills = []
    m.reap_terminal_workers(conn, signal_fn=_signal_fn(pids, kills))
    assert kills == [], "a post-close heartbeat defers the reap"
    row = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] == LAUNCHER_PID, "evidence kept while deferring"

    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE run_id=? AND kind='heartbeat'",
            (ENDED_AT - 10, run_id),
        )

    m.reap_terminal_workers(conn, signal_fn=_signal_fn(pids, kills))
    assert [pid for pid, _ in kills] == [LAUNCHER_PID], \
        "no post-close liveness: the reap proceeds without the worker_result event"
    row = conn.execute("SELECT worker_pid FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] is None


def test_guard_still_yields_to_worker_result_event(board, monkeypatch):
    """A recorded worker_result event ends the finishing wait immediately,
    heartbeat or not (the guarantee is unchanged)."""
    conn, tid, run_id, _pids = _seed_completed_run(
        board, monkeypatch, launcher_alive=True, bound_alive=False, patch_finishing=False)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'worker_result', '{}', ?)",
            (tid, run_id, ENDED_AT + 1),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'heartbeat', NULL, ?)",
            (tid, run_id, ENDED_AT + 10),
        )
    assert m._worker_still_finishing(conn, {"id": run_id, "ended_at": ENDED_AT}) is False
