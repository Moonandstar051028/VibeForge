"""Observe one current campaign, its owned processes, GPU and actual Agent actions."""
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

from vf_runtime.engine.process import process_identity, same_process
from vf_runtime.observability.logs import append_record
from vf_runtime.storage.atomic import atomic_write, write_json


def read(path, default=None):
    try:return json.loads(path.read_text())
    except (OSError, ValueError):return default


def records(path):
    try:lines=path.read_text(errors='replace').splitlines()
    except OSError:return []
    result=[]
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:continue
    return result


def latest_request(run, folder):
    paths=[run/'initial/engine.log']
    if folder:paths+=list((folder/'logs').glob('*.log'))
    for path in sorted((p for p in paths if p.exists()),key=lambda p:p.stat().st_mtime,reverse=True):
        with path.open('rb') as f:
            # Long generations emit enough progress output to push the last
            # completed request outside a 128 KiB tail. Keep it visible while
            # the next request is still running.
            f.seek(0,2);size=f.tell();f.seek(max(0,size-2097152));lines=f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            if not line.startswith('{'):continue
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('event')=='request_observation':
                keys=['operation','backend_wall_seconds','input_tokens','output_tokens','finish_reason',
                      'reasoning_tokens','visible_tokens','thinking_complete','output_tokens_per_second','cuda_memory']
                return {'log':str(path),**{k:d.get(k) for k in keys}}
    return None


def observe(run, output, gpu_uuid, stage_since):
    state=read(run/'state.json',{})
    active=read(output.parent/'active.json',{})
    launch=read(output/'launch.json',{})
    supervisor=read(output/'supervisor-status.json',{})
    objective=read(run/'config.json',{}).get('evaluation',{}).get('objective')
    controller_alive=bool(launch.get('identity') and same_process(launch['identity']))
    number=state.get('current_round');folder=run/'rounds'/f'{number:04d}' if isinstance(number,int) else None
    tools=records(folder/'agent/tools.jsonl') if folder else []
    shell=[d for d in tools if 'command' in d and 'exit_code' in d]
    normalized=[' '.join(d['command'].split()) for d in shell]
    counts=Counter(normalized)
    changed=next((i+1 for i,d in enumerate(tools) if d.get('candidate_check',{}).get('runtime_changed_files')),None)
    errors=[{'sequence':i+1,'exit_code':d.get('exit_code'),'timed_out':d.get('timed_out'),
             'command':d['command'][:500],'output_tail':d.get('output','')[-800:]}
            for i,d in enumerate(shell) if d.get('exit_code') not in (0,None)]
    conversation=read(folder/'agent/conversation.json',[]) if folder else []
    last=conversation[-1] if conversation else {}
    health=None;owned=read(run/'state/active.engine.json',{})
    if owned and same_process(owned):
        try:
            url=owned.get('url') or f"http://127.0.0.1:{owned['port']}"
            with urlopen(url.rstrip('/')+'/health',timeout=2) as response:d=json.load(response)
            if d.get('instance_token')==owned.get('token'):
                runtime=d.get('runtime',{})
                health={'ready':d.get('ready'),'max_context':d.get('max_context'),'device':d.get('device'),
                        'model_class':runtime.get('model_class'),'model_dtype':runtime.get('model_dtype'),
                        'model_device':runtime.get('model_device'),'pid':owned['pid']}
        except Exception as e:health={'ready':False,'error':str(e),'pid':owned['pid']}
    command=['nvidia-smi','--id='+gpu_uuid,'--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits']
    gpu={}
    try:
        d=subprocess.run(command,capture_output=True,text=True,timeout=5,check=True).stdout.strip().split(', ')
        gpu=dict(zip(['uuid','name','memory_used_mib','memory_total_mib','utilization_percent','temperature_c','power_w'],d))
    except Exception as e:gpu={'error':str(e)}
    alerts=[]
    if len(shell)>=8 and changed is None:alerts.append('Agent has run at least eight Shell commands without a runtime file change')
    if any(n>=3 for n in counts.values()):alerts.append('An identical normalized Shell command appeared at least three times; inspect whether outputs changed')
    if tools and tools[-1].get('candidate_check',{}).get('errors'):alerts.append('Latest candidate static check has errors')
    if any(d.get('timed_out') for d in shell):alerts.append('A Shell command timed out')
    if state.get('integrity_failed'):alerts.append('Experiment integrity failure; automatic recovery must remain blocked')
    if supervisor.get('phase')=='needs_diagnosis':alerts.append('Supervisor requires diagnosis')
    if time.time()-stage_since>1800:alerts.append('Current stage has lasted over 1800 seconds; inspect progress and deadlines')
    req=latest_request(run,folder)
    if req and req.get('finish_reason')=='incomplete_thinking':alerts.append('Latest generation exhausted output before thinking ended')
    decisions=[read(p) for p in sorted((run/'rounds').glob('*/decision.json'))]
    pipeline_pid=active.get('pipeline_pid') or supervisor.get('pipeline_pid')
    return {'updated_at':time.time(),'run':str(run),'controller_alive':controller_alive,
            'performance_objective':objective,
            'pipeline_pid':pipeline_pid,'pipeline_process':process_identity(pipeline_pid) if pipeline_pid else None,
            'supervisor':supervisor,'state':state,'stage_elapsed_seconds':round(time.time()-stage_since,1),
            'gpu':gpu,'engine_health':health,'last_request':req,
            'agent':{'tool_calls':len(tools),'shell_calls':len(shell),'first_runtime_change_call':changed,
                     'edit_calls':sum(d.get('tool')=='edit_file' for d in tools),
                     'last_tool':{k:tools[-1].get(k) for k in ('tool','passed','error','candidate_check')} if tools else None,
                     'duplicate_command_count':len(normalized)-len(counts),'failed_commands':errors[-5:],
                     'last_commands':[{'command':d['command'][:1200],'exit_code':d.get('exit_code'),
                                      'runtime_changed_files':d.get('candidate_check',{}).get('runtime_changed_files'),
                                      'check_errors':d.get('candidate_check',{}).get('errors',[])} for d in shell[-4:]],
                     'last_message_role':last.get('role'),'last_message_tail':str(last.get('content',''))[-800:],
                     'context_recoveries':len(read(folder/'agent/context-restarts.json',[])) if folder else 0},
            'decisions':decisions,'alerts':alerts}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--interval',type=float,default=10);a=p.parse_args()
    stopping=False
    def stop(*args):
        nonlocal stopping;stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    stage=None;stage_since=time.time();seen_alerts=set()
    while not stopping:
        state=read(a.run/'state.json',{});current=(state.get('current_round'),state.get('stage'))
        if current!=stage:stage=current;stage_since=time.time();seen_alerts.clear()
        try:
            result=observe(a.run,a.workspace,a.gpu_uuid,stage_since)
            from review_agent_context import review
            review(result,a.workspace)
            write_json(a.workspace/'live-status.json',result)
            append_record(a.workspace/'telemetry.jsonl',{k:result[k] for k in ['updated_at','controller_alive','state','gpu','engine_health','alerts']})
            for alert in result['alerts']:
                if alert not in seen_alerts:
                    append_record(a.workspace/'alerts.jsonl',{'at':time.time(),'round':state.get('current_round'),'stage':state.get('stage'),'alert':alert});seen_alerts.add(alert)
            rows=['# 当前运行监控','',f"更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}",f"运行：{a.run}",
                  f"控制进程存活：{result['controller_alive']}",f"状态：{state.get('status')} / {state.get('stage')}；轮次 {state.get('current_round')}；完成 {state.get('completed_rounds',0)}",'',
                  f"性能准入主指标：{result['performance_objective']}",
                  '单请求端到端 token/s = 每个请求的输出 token 数 / 客户端总耗时（含预填充、服务端排队和生成），取请求均值，再按场景权重取几何平均。',
                  f"GPU：{result['gpu']}",f"模型服务：{result['engine_health']}",
                  f"本轮 Shell {result['agent']['shell_calls']} 次；首次运行时代码修改：{result['agent']['first_runtime_change_call']}；重复命令 {result['agent']['duplicate_command_count']} 次",'',
                  f"最近 Agent 请求证据：{result['agent'].get('last_request_trace')}；显式思考字符 {result['agent'].get('reasoning_chars',0)}；输入中 omit 工具结果 {result['agent'].get('omitted_tool_results',0)}",'',
                  '## 待检查信号','',*(['- '+s for s in result['alerts']] or ['- 暂无规则触发信号；不代表候选已通过评估。']),'',
                  '## 最近命令','',*[f"- exit={d['exit_code']} {d['command'][:350]}" for d in result['agent']['last_commands']],'',
                  '## 宿主轮次判定','',*[f"- {d.get('round')}: {d.get('outcome')} / {d.get('reason')} ({d.get('stage')})" for d in result['decisions']],'']
            atomic_write(a.workspace/'LIVE.md','\n'.join(rows))
            if not result['controller_alive']:break
        except Exception as e:
            append_record(a.workspace/'monitor-errors.jsonl',{'at':time.time(),'error':repr(e)})
        time.sleep(a.interval)


if __name__=='__main__':main()
