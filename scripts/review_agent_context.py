"""Automatically preserve actual context and reasoning evidence at anomaly signals."""
import json
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, write_json


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def review(result, output):
    state = result['state']
    number = state.get('current_round')
    if not isinstance(number, int):
        return
    folder = Path(result['run'])/'rounds'/f'{number:04d}'/'agent'
    requests = sorted((folder/'requests').glob('[0-9]*'))
    if not requests:
        return
    latest = requests[-1]
    response = read(latest/'response.json', {})
    args = read(latest/'request.json', {}).get('arguments', {})
    messages = args.get('messages', [])
    reasoning = response.get('reasoning_content', '')
    result['agent'].update(last_request_trace=str(latest), reasoning_chars=len(reasoning),
                          traced_request_status=response.get('status'),
                          input_tool_results=sum(d.get('role')=='tool' for d in messages),
                          omitted_tool_results=sum(d.get('role')=='tool' and d.get('content')=='omit' for d in messages))
    if result['agent']['duplicate_command_count']:
        result['alerts'].append('Repeated command observed; inspect saved actual input and reasoning evidence')
    if result['agent']['context_recoveries']:
        result['alerts'].append('Context overflow recovery occurred; inspect exact context before and after folding')
    if response.get('status')=='completed' and 'length' in response.get('finish_reasons', []):
        result['alerts'].append('Model reached its output budget; check whether thinking and tool call completed')
    if result['agent']['failed_commands']:
        result['alerts'].append('A tool returned a nonzero exit code; inspect feedback and next action')
    if not result['alerts']:
        return
    target = output/'context-reviews'/f"round-{number:04d}-request-{latest.name}-{response.get('status','unknown')}"
    if target.exists():
        return
    target.mkdir(parents=True)
    # Immutable copies prevent later streaming updates from changing anomaly evidence.
    for name in ('request.json', 'response.json', 'before-fold.json'):
        original = latest/name
        if original.exists():
            atomic_write(target/name, original.read_text())
    before = read(target/'before-fold.json', [])
    report = {'at':result['updated_at'], 'round':number, 'signals':result['alerts'],
              'shell_calls':result['agent']['shell_calls'], 'trace':str(latest),
              'request_status':response.get('status'), 'reasoning_chars':len(reasoning),
              'before_fold_messages':len(before), 'actual_adapter_messages':len(messages),
              'input_tool_results':result['agent']['input_tool_results'],
              'omitted_tool_results':result['agent']['omitted_tool_results'],
              'finish_reasons':response.get('finish_reasons', []),
              'scope':'Automatic evidence checks, not a judgment that the model reasoning is mathematically correct. '
                      'Request files contain actual adapter inputs; complete reasoning and output are in response.json.'}
    write_json(target/'review.json', report)
    atomic_write(output/'CONTEXT-REVIEW.md', '\n'.join([
        '# 最近异常的上下文与思考核查', '', f'轮次：{number}；Shell 调用：{report["shell_calls"]}',
        f'证据目录：{target}', f'请求状态：{report["request_status"]}；已记录显式思考字符：{len(reasoning)}',
        f'实际适配层输入的工具结果：{report["input_tool_results"]}；其中 omit：{report["omitted_tool_results"]}', '',
        *['- '+s for s in result['alerts']], '',
        'before-fold.json：框架裁剪前输入；request.json：裁剪与原生接口转换后的实际请求参数；response.json：模型返回的思考、工具调用、输出和状态。',
        '自动核查保存证据并检测结构性异常，不证明思考正确；解释推理和代码是否成立仍须结合源码及真实验证。', '']))
