"""Read-only observation of the owned continuous run; writes only workspace evidence."""
import json
import sys
import time
from pathlib import Path

from vf_runtime.storage.atomic import write_json
from vf_runtime.observability.logs import append_record

run, output, pid = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
state_key, stage_since, previous_stage = None, time.time(), None
seen_tools, seen_decisions, seen_connection = set(), set(), set()
seen_recoveries = set()

def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def event(kind, **data):
    append_record(output/'watch-events.jsonl', {'at':time.time(),'kind':kind,**data})

def alive():
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return b'vibeforge_v2' in command and str(run).encode() in command
    except OSError:
        return False

while True:
    active=alive()
    state=read(run/'state.json',{})
    key=tuple(state.get(k) for k in ('status','stage','current_round','completed_rounds','stable','integrity_failed'))
    if key!=state_key:
        event('state',state=state)
        state_key=key
    current=(state.get('current_round'),state.get('stage'))
    if current!=previous_stage:
        stage_since=time.time();previous_stage=current
    for folder in sorted((run/'rounds').glob('[0-9]*')):
        number=folder.name
        connection=read(folder/'connection/connection.json')
        if connection and number not in seen_connection:
            event('connection',round=number,report=connection);seen_connection.add(number)
        tools=folder/'agent/tools.jsonl'
        if tools.exists():
            for index,line in enumerate(tools.read_text(errors='replace').splitlines()):
                if (number,index) in seen_tools:continue
                try:r=json.loads(line)
                except ValueError:continue
                check=r.get('candidate_check',{})
                event('agent_tool',round=number,sequence=index+1,command=r.get('command'),
                      tool=r.get('tool','shell'),arguments=r.get('arguments'),exit_code=r.get('exit_code'),
                      timed_out=r.get('timed_out'),elapsed_seconds=r.get('elapsed_seconds'),
                      source_hash=check.get('source_hash'),ready=check.get('ready_for_host_evaluation'),
                      check_errors=check.get('errors',[]),check_warnings=check.get('warnings',[]),
                      output_tail=r.get('output','')[-1600:] if r.get('exit_code') not in (0,None) else None)
                seen_tools.add((number,index))
        for index, recovery in enumerate(read(folder/'agent/context-restarts.json', [])):
            if (number,index) not in seen_recoveries:
                event('official_context_recovery',round=number,report=recovery)
                seen_recoveries.add((number,index))
        decision=read(folder/'decision.json')
        if decision and number not in seen_decisions:
            config=read(folder/'engine/engine.json',{})
            event('decision',round=number,decision=decision,
                  candidate_options={k:config.get(k) for k in ('attn_implementation','request_scheduler',
                     'continuous_batching','microbatch_max_size','microbatch_wait_ms')})
            quality=read(folder/'quality/report.json',{})
            performance=read(folder/'diagnostic-performance/report.json',{})
            event('evaluation',round=number,quality_passed=quality.get('passed'),
                  quality_metrics=quality.get('metrics'),paired_quality=quality.get('paired'),
                  exploratory_valid=performance.get('valid'),exploratory_gain=performance.get('gain'),
                  performance_profiles=(performance.get('comparison') or {}).get('profiles'))
            seen_decisions.add(number)
    write_json(output/'watch-status.json',{'updated_at':time.time(),'pipeline_pid':pid,'process_alive':active,
        'state':state,'stage_elapsed_seconds':time.time()-stage_since,'observed_tool_calls':len(seen_tools),
        'observed_decisions':len(seen_decisions),'context_recoveries':len(seen_recoveries),
        'current_engine':{key:value for key,value in read(run/'current/manifest.json',{}).items()
                          if key in ('role','source','source_hash','runtime_observed_round')}})
    if not active:
        event('process_exit',state=state)
        break
    time.sleep(10)
