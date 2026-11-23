"""Host-computed bottleneck digest: derived ratios from saved measurements, never new numbers.

The raw survey is large and mostly unlabeled; a mid-sized coding model reliably
reads tables of counters but rarely derives ratios such as "GPU busy fraction
during decode" by itself. This module computes those ratios once, states the
arithmetic, and names the code location that the ratio points at. It does not
decide what the Agent must implement.
"""
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write

# Nominal HBM bandwidth (bytes/s) by GPU marketing name; only used for a
# labelled lower-bound estimate. Unknown devices simply omit the estimate.
NOMINAL_BANDWIDTH = {'A800 80GB PCIE': 1.935e12, 'A800': 1.935e12, 'A100 80GB PCIE': 1.935e12,
                     'A100-SXM4-80GB': 2.039e12, 'A100': 1.555e12, 'H800 PCIE': 2.0e12,
                     'H100 PCIE': 2.0e12, 'H800': 3.35e12, 'H100': 3.35e12, 'L40S': 0.864e12,
                     '4090': 1.008e12}


def _phase(record, name):
    for row in record.get('phases') or []:
        if row.get('name') == name:
            return row
    return {}


def _operator_calls(record, name):
    value = (record.get('measurements') or {}).get(f'operator.{name}.calls') or {}
    return value.get('value')


def active_parameters(model):
    """Per-token weight elements touched by one decode step (approximate, labelled as such)."""
    config = model.get('text_config') or model
    try:
        hidden, layers = config['hidden_size'], config['num_hidden_layers']
        heads, kv_heads = config['num_attention_heads'], config.get('num_key_value_heads', config['num_attention_heads'])
        head_dim = config.get('head_dim') or hidden // heads
        vocab = config['vocab_size']
    except (KeyError, TypeError):
        return None
    attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    experts, top_k = config.get('num_experts'), config.get('num_experts_per_tok')
    if experts and top_k and config.get('moe_intermediate_size'):
        mlp = top_k * 3 * hidden * config['moe_intermediate_size'] + hidden * experts
        kind = f'稀疏前馈结构（每层 {experts} 个专家子网络，每 token 激活 {top_k} 个，单个宽度 {config["moe_intermediate_size"]}）'
    else:
        mlp = 3 * hidden * config.get('intermediate_size', 4 * hidden)
        kind = '稠密前馈结构'
    return {'elements': layers * (attention + mlp) + hidden * vocab, 'layers': layers, 'kind': kind,
            'experts': experts, 'top_k': top_k}


def _bandwidth(config):
    name = (config.get('hardware') or {}).get('name_contains', '') or ''
    description = (config.get('hardware') or {}).get('description', '') or ''
    text = (name + ' ' + description).upper()
    for key in sorted(NOMINAL_BANDWIDTH, key=len, reverse=True):
        if key in text:
            return key, NOMINAL_BANDWIDTH[key]
    return None, None


def digest(observations, profile, config):
    """Return (markdown, facts). Facts are numbers already measured or simple arithmetic on them."""
    lines = ['# 宿主瓶颈摘要（由已保存的测量数据计算，非新测量）', '']
    facts = {'profiles': {}}
    model = (profile or {}).get('model') or {}
    params = active_parameters(model)
    gpu, bandwidth = _bandwidth(config)
    dtype_bytes = 2 if (config.get('precision') or {}).get('weight_dtype', 'bfloat16') in ('bfloat16', 'float16') else 4
    floor = params['elements'] * dtype_bytes / bandwidth if params and bandwidth else None
    if params:
        facts['active_weight_bytes_per_token'] = params['elements'] * dtype_bytes
        lines.append(f"模型结构：{params['layers']} 层，{params['kind']}。单 token decode 约读取 "
                     f"{params['elements'] * dtype_bytes / 1e9:.1f} GB 激活权重（含 lm_head，近似值）。")
    if floor:
        facts['bandwidth_floor_tpot_seconds'] = floor
        lines.append(f'{gpu} 名义显存带宽 {bandwidth / 1e12:.2f} TB/s → batch=1 decode 的带宽下限约 '
                     f'{floor * 1000:.1f} ms/token（理论下界，不可达，只用于判断离上限多远）。')
    lines.append('')
    lines.append('| 场景 | 输入 tokens | TPOT ms | TTFT s | decode 设备/CPU 累计时间比（非利用率） | 每输出 token 的 kernel launch | 每输出 token 的 host 同步 | 前馈与 attention 的 CPU 时间比 |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for record in observations.get('profiles') or []:
        normal = record.get('normal_request') or {}
        decode = _phase(record, 'vf_runtime.decode')
        mlp = _phase(record, 'vf_runtime.module.mlp')
        attention = _phase(record, 'vf_runtime.module.full_attention') or _phase(record, 'vf_runtime.module.attention')
        outputs = record.get('output_tokens') or normal.get('output_tokens')
        busy = (decode.get('device_total_seconds') / decode['cpu_total_seconds']
                if decode.get('cpu_total_seconds') and decode.get('device_total_seconds') is not None else None)
        launches = (record.get('diagnostics') or {}).get('kernel_launches_per_output_token')
        syncs = _operator_calls(record, 'cudaStreamSynchronize')
        syncs_per_token = syncs / outputs if syncs and outputs else None
        ratio = (mlp['cpu_total_seconds'] / attention['cpu_total_seconds']
                 if mlp.get('cpu_total_seconds') and attention.get('cpu_total_seconds') else None)
        row = {'input_tokens': normal.get('input_tokens'), 'tpot_seconds': normal.get('tpot_seconds'),
               'ttft_seconds': normal.get('ttft_seconds'), 'decode_gpu_busy_fraction': busy,
               'kernel_launches_per_token': launches, 'host_syncs_per_token': syncs_per_token,
               'mlp_to_attention_cpu_ratio': ratio}
        facts['profiles'][record.get('profile_id', '?')] = row
        fmt = lambda value, scale=1, digits=1: '—' if value is None else f'{value * scale:.{digits}f}'
        lines.append(f"| {record.get('profile_id')} | {row['input_tokens']} | {fmt(row['tpot_seconds'], 1000)} | "
                     f"{fmt(row['ttft_seconds'], 1, 2)} | {fmt(busy, 100, 0)}% | {fmt(launches, 1, 0)} | "
                     f"{fmt(syncs_per_token, 1, 0)} | {fmt(ratio)} |")
    rows = list(facts['profiles'].values())
    def mean(key):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return sum(values) / len(values) if values else None
    tpot, busy, launches = mean('tpot_seconds'), mean('decode_gpu_busy_fraction'), mean('kernel_launches_per_token')
    syncs, ratio = mean('host_syncs_per_token'), mean('mlp_to_attention_cpu_ratio')
    lines += ['', '## 由上表直接得出的判断（每条都给出所依据的数字）', '']
    findings = []
    if tpot and floor:
        facts['tpot_to_bandwidth_floor'] = tpot / floor
        findings.append(f'当前 TPOT {tpot * 1000:.0f} ms，与估算权重传输时间的比值为 {tpot / floor:.1f}。'
                        '该估算假定每步读取激活权重并使用名义带宽，不能据此计算实测时间组成或预期加速比例。')
    if busy is not None:
        findings.append(f'decode 设备/CPU 累计时间比约 {busy:.2f}。profiler 累计时间可能有重叠，'
                        '这个比值不是 GPU 利用率，其补数也不是 CPU 发射空隙。'
                        'host/launch-bound 只能作为候选假设，需结合时间线、同步调用及局部实验验证。')
    if launches:
        per_layer = launches / params['layers'] if params else None
        findings.append(f'每个输出 token 约 {launches:.0f} 次 kernel launch'
                        + (f'（每层约 {per_layer:.0f} 次）' if per_layer else '') + '。')
    if syncs:
        findings.append(f'每个输出 token 约 {syncs:.0f} 次 cudaStreamSynchronize；同步通常来自 nonzero/where/.item()/'
                        '以 GPU 张量做 Python 循环或索引，会阻断 CPU 提前排队。')
    if ratio and ratio > 2:
        findings.append(f'前馈（MLP）模块的 CPU 时间约为 attention 的 {ratio:.1f} 倍。')
    total_bytes = sum((model.get('dtype_bytes') or {}).values()) if model else 0
    if total_bytes:
        facts['total_weight_bytes'] = total_bytes
        large = total_bytes > 40e9
        findings.append(f'全部权重约 {total_bytes / 1e9:.0f} GB' + ('，已占 GPU 显存的大部分' if large else '')
                        + '：复制或重排权重的改动必须释放或复用原张量（harness 的 parameter_memory_ratio 应约等于 1）。')
        findings.append('若热点模块内部按子模块做 Python 循环或数据相关的分支（可在 observations.json 的 '
                        'implementation_excerpts 中查看 forward 定义；装饰器可能再分发到其它实现），kernel 数与同步数会随循环次数增长；'
                        'harness 的 FLOPs 与子模块调用次数用于提示额外计算风险；最终性能由 GPU 测量决定。')
    records = observations.get('profiles') or []
    locations = ((records[0].get('implementation_index') or {}).get('locations') or {}) if records else {}
    hot = [(name, where) for name, where in locations.items()
           if any(word in name.lower() for word in ('moe', 'mlp.forward', 'attention.forward', 'backend.generate', 'block.forward'))
           and isinstance(where, dict) and where.get('path')]
    if hot:
        findings.append('已加载的源码定义位置（不代表这些定义都被本次请求调用；装饰器入口见 implementation_index 的 callable_location。'
                        '只读参考；site-packages 文件是共享环境，不得直接修改，'
                        '请在候选目录实现并在 engine.py 中替换 forward 或子类化）：'
                        + '；'.join(f"{name} → {where['path']}:{where.get('line')}" for name, where in hot[:5]))
    if not findings:
        findings.append('未能从现有字段推出明确结论；请直接查看 observations.json。')
    lines += [f'{index}. {text}' for index, text in enumerate(findings, 1)]
    lines += ['', '数字来源：rounds/NNNN/observations/*.json 的 normal_request、phases、diagnostics、measurements 字段。'
              '比例是同一 profiler 运行内的近似值，不是正式性能。', '']
    facts['findings'] = findings
    return '\n'.join(lines), facts


def write_digest(observations, profile, config, folder):
    text, facts = digest(observations, profile, config)
    atomic_write(Path(folder) / 'digest.md', text)
    return text, facts
