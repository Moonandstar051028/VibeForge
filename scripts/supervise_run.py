"""Keep an owned continuous run alive; checkpoint recovery, never edit engine code."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from vf_runtime.engine.process import process_identity, stop_record, startup_diagnostic
from vf_runtime.observability.logs import append_record
from vf_runtime.storage.atomic import read_json, write_json
from vf_runtime.versioning.artifacts import source_hash
from vibeforge_v2.environment import Environments


def recovery_check(run, seal):
    """Never reset an integrity failure or resume against changed independent inputs."""
    state = read_json(run / 'state.json')
    if state.get('integrity_failed'):
        raise RuntimeError('Pipeline reported an integrity failure; diagnosis is required')
    for name, expected in seal.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Recovery input changed: ' + name)
    stable = (run / state['stable']).resolve()
    stable.relative_to(run.resolve())
    if source_hash(stable) != state['stable_hash']:
        raise RuntimeError('Stable checkpoint source identity mismatch')
    config = read_json(run / 'config.json')
    Environments(config, run).verify(stable)
    return state


def recovery_seal(run, project):
    inputs = [run / name for name in ('config.json', 'cases.json', 'reference.json',
                                     'baseline-quality.json', 'agent-prompt.md')]
    inputs += list((project / 'src').rglob('*.py'))
    inputs += [p for p in (run / 'evaluation-inputs').rglob('*') if p.is_file()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


def supervise(run, output, *, interval=10, retry_delay=15):
    run, output = run.resolve(), output.resolve()
    project = Path(read_json(run / 'config.json')['project'])
    active_path = output.parent / 'active.json'
    active = read_json(active_path)
    if Path(active['run']).resolve() != run:
        raise RuntimeError('Active run does not match requested run')
    old_pid = active.get('pipeline_pid')
    if old_pid and Path(f'/proc/{old_pid}/cmdline').exists():
        command = Path(f'/proc/{old_pid}/cmdline').read_bytes()
        if b'vibeforge_v2' in command and str(run).encode() in command:
            raise RuntimeError('An owned pipeline is already running')
    seal_path = output / 'supervisor-seal.json'
    if not seal_path.exists():
        write_json(seal_path, recovery_seal(run, project))
    seal = read_json(seal_path)
    stopping = False
    child, watcher = None, None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def event(kind, **data):
        append_record(output / 'operations.jsonl', {'at': time.time(), 'kind': kind, **data})

    def status(phase, **data):
        write_json(output / 'supervisor-status.json', {
            'at': time.time(), 'supervisor': process_identity(os.getpid()), 'phase': phase,
            'pipeline_pid': child.pid if child is not None else None, **data})

    command = [sys.executable, '-u', '-m', 'vibeforge_v2', 'resume', '--output', str(run), '--continuous']
    progress, failures = None, 0
    try:
        while not stopping:
            try:
                state = recovery_check(run, seal)
                # A dead pipeline may have left its separately owned engine alive.
                cleaned = stop_record(run / 'state/active.engine.json')
                event('checkpoint_recovery', cleaned_owned_engine=cleaned,
                      completed_rounds=state['completed_rounds'], stable=state['stable'])
            except Exception as error:
                event('recovery_blocked', error=f'{type(error).__name__}: {error}')
                status('needs_diagnosis', error=str(error))
                return 2
            with (output / 'run.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ), start_new_session=True)
            identity = process_identity(child.pid)
            active.update(at=time.time(), pipeline_pid=child.pid, pipeline_identity=identity,
                          supervisor_pid=os.getpid(), command=command, candidate_quality_repeats=1)
            with (output / 'watcher.log').open('ab', buffering=0) as log:
                watcher = subprocess.Popen([sys.executable, '-u', str(project / 'scripts/watch_run.py'),
                    str(run), str(output), str(child.pid)], cwd=project, env=dict(os.environ),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active['watcher_pid'] = watcher.pid
            write_json(active_path, active)
            event('pipeline_resumed', pid=child.pid, command=command, continuous=True)
            while child.poll() is None:
                if stopping:
                    child.send_signal(signal.SIGINT)
                status('stopping' if stopping else 'monitoring')
                time.sleep(min(interval, 1) if stopping else interval)
            if watcher.poll() is None:
                # Let the read-only watcher record the process exit before replacing it.
                try:
                    watcher.wait(timeout=interval + 5)
                except subprocess.TimeoutExpired:
                    watcher.terminate()
                    watcher.wait(timeout=5)
            if stopping:
                status('stopped_by_operator', returncode=child.returncode)
                return 0
            state = read_json(run / 'state.json')
            event('unexpected_pipeline_exit', returncode=child.returncode, state=state,
                  error_tail=startup_diagnostic(output / 'run.log'))
            if child.returncode == 0 and state.get('status') == 'completed':
                status('completed')
                return 0
            checkpoint = state.get('completed_rounds')
            failures = failures + 1 if checkpoint == progress else 1
            progress = checkpoint
            if failures >= 3:
                status('needs_diagnosis', error='Three exits without checkpoint progress')
                event('recovery_blocked', error='Repeated exits without checkpoint progress')
                return 2
            # Cleanup and checkpoint replay repair process loss, not arbitrary code defects.
            status('recovering', failed_attempts_without_progress=failures)
            deadline = time.monotonic() + retry_delay
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        status('stopped_by_operator')
        return 0
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait(timeout=60)
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
            watcher.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--retry-delay', type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0 or args.retry_delay < 0:
        parser.error('interval must be positive and retry-delay must be nonnegative')
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise(args.run, args.workspace, interval=args.interval, retry_delay=args.retry_delay)


if __name__ == '__main__':
    sys.exit(main())
