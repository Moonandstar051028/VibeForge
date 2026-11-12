"""Independent final-versus-initial comparison using the frozen admission protocol."""

import time
from vibeforge.controller.admission import decide
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import read_json, write_json, digest
from vibeforge.storage.locks import exclusive_lock
from vibeforge.observability.reports import write_campaign_report
from vibeforge.profiling.collector import collect


def compare_initial(paths, campaign_id):
    controller = Supervisor(paths, campaign_id)
    try:
        with exclusive_lock(paths.data / 'state' / f'{campaign_id}.lock'):
            with exclusive_lock(paths.data / 'state' / 'device.lock'):
                try:
                    state = controller.database.get(campaign_id)
                    if state['state'] not in {'STOPPED', 'HALTED'} or not state.get('initial_ref') or not state.get('stable_ref'):
                        raise ValueError('Comparison requires a stopped campaign with initial and stable releases')
                    controller.config = read_json(controller.directory / 'manifest.json')
                    if controller.config['evaluation'].get('measurement_protocol_version') != 2:
                        raise ValueError('Create a new campaign with comparable formal timing before comparison')
                    if controller.config['config_hash'] != digest({k: v for k, v in controller.config.items() if k != 'config_hash'}):
                        raise ValueError('Frozen campaign configuration was modified')
                    profile, preflight = collect(controller.config)
                    if not preflight['passed'] or profile['profile_hash'] != state['profile_hash']:
                        raise ValueError('Comparison environment differs from the campaign or preflight failed')
                    folder = controller.directory / 'comparisons' / str(time.time_ns())
                    folder.mkdir(parents=True)
                    # This explicitly requested comparison has a separate bounded evaluation
                    # budget; the completed campaign's counters and start time stay intact.
                    controller.evaluation_deadline = time.monotonic() + controller.config['budget']['max_seconds']
                    baseline, candidate = controller.benchmark(state['initial_ref'], state['stable_ref'], folder)
                    result = decide({'passed': True}, baseline, candidate, controller.config['evaluation']).to_dict()
                    record = {
                        'initial_ref': state['initial_ref'], 'stable_ref': state['stable_ref'],
                        'profile_hash': profile['profile_hash'], 'config_hash': controller.config['config_hash'],
                        'decision': result, 'baseline': baseline, 'candidate': candidate,
                        'evidence_path': str(folder.relative_to(controller.directory)),
                        'scope': 'Fresh paired final-versus-initial measurement; no version promotion. '
                                 'Output references are checked on every timed request; release correctness and quality come from campaign gates.',
                    }
                    write_json(folder / 'report.json', record)
                    write_json(controller.directory / 'cumulative-comparison.json', record)
                    return record
                finally:
                    if controller.engine:
                        controller.stop_engine()
                    write_campaign_report(controller.database, paths, campaign_id)
    finally:
        controller.close()
