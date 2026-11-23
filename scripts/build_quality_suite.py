"""Reproducible local diagnostic suite; not a public benchmark leaderboard."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cases = [{**row, 'category': 'core'} for row in json.loads((ROOT / 'benchmarks/task-quality-core.json').read_text())]
    def add(name, category, prompt, answer, **extra):
        cases.append({'id': name, 'category': category, 'mode': 'chat', 'prompt': prompt,
                      'expected': str(answer), 'match': 'exact', 'max_new_tokens': 48, **extra})
    for i in range(20):
        a, b = 13 + i * 3, 7 + i
        add(f'multiply_{i}', 'arithmetic', f'只输出整数结果：{a} × {b}', a * b)
        add(f'expression_{i}', 'arithmetic', f'只输出整数结果：({a} + {b}) × 3 - {i}', (a + b) * 3 - i)
        add(f'divide_{i}', 'arithmetic', f'只输出整数结果：{a * b} ÷ {b}', a)
        value = f'Vf{i:02d}-Sky'
        add(f'reverse_{i}', 'string', f'将下面 ASCII 字符串逐字符反转，只输出反转结果：{value}', value[::-1])
        # Different contexts and distractors; label locations alternate.
        records = [f'item_{j}: {i * 37 + j * 11}' for j in range(8)]
        target = i % 8
        add(f'read_{i}', 'reading', '\n'.join(records) + f'\n只输出 item_{target} 的数值。', i * 37 + target * 11)
        expression = f'({a} > {b}) and ({i} % 2 == 0)'
        add(f'logic_{i}', 'logic', f'计算 Python 表达式 {expression}，只输出 True 或 False。', i % 2 == 0)
    functions = [
        ('sum_even', '返回 xs 中所有偶数的和。', 'assert sum_even([1,2,3,4]) == 6\nassert sum_even([]) == 0\nassert sum_even([-2,0,5]) == -2'),
        ('unique_order', '返回去重后的列表，保持首次出现顺序，元素为整数。', 'assert unique_order([3,1,3,2,1]) == [3,1,2]\nassert unique_order([]) == []'),
        ('is_prime', '判断整数 n 是否为素数，返回 bool。', 'assert is_prime(2) is True\nassert is_prime(1) is False\nassert is_prime(49) is False\nassert is_prime(97) is True'),
        ('count_words', '参数 s 为字符串，按空白分词，返回词数。', 'assert count_words(" a  b\\nc ") == 3\nassert count_words("") == 0'),
        ('rotate', '参数 xs 为列表、k 为非负整数，返回向右旋转 k 次的新列表。', 'assert rotate([1,2,3],4) == [3,1,2]\nassert rotate([],2) == []\nassert rotate([1],0) == [1]'),
        ('flatten', '参数 xss 为二维列表，只展开一层。', 'assert flatten([[1,2],[],[3]]) == [1,2,3]\nassert flatten([]) == []'),
        ('balanced', '参数 s 只包含圆括号，判断括号是否平衡，返回 bool。', 'assert balanced("(()())") is True\nassert balanced(")(") is False\nassert balanced("") is True'),
        ('merge_sorted', '参数 a,b 为已升序排列的整数列表，返回合并后的升序列表。', 'assert merge_sorted([1,3],[2,3]) == [1,2,3,3]\nassert merge_sorted([],[-1]) == [-1]'),
        ('gcd', '参数 a,b 为非负整数，返回最大公约数，约定 gcd(0,0)=0。', 'assert gcd(54,24) == 6\nassert gcd(0,7) == 7\nassert gcd(0,0) == 0'),
        ('run_lengths', '参数 s 为字符串，返回连续相同字符的 (字符,次数) 元组列表。', 'assert run_lengths("aaabbc") == [("a",3),("b",2),("c",1)]\nassert run_lengths("") == []'),
        ('first_missing', '参数 xs 为整数列表，返回没有出现的最小正整数。', 'assert first_missing([3,4,-1,1]) == 2\nassert first_missing([]) == 1\nassert first_missing([1,2,3]) == 4'),
        ('transpose', '参数 rows 为矩形二维列表，返回转置后的二维列表，空列表返回空列表。', 'assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\nassert transpose([]) == []'),
    ]
    for name, requirement, tests in functions:
        add('code_' + name, 'coding', f'只输出 Python 函数定义，函数名必须为 {name}。仅使用标准库。{requirement}', '',
            match='python', entry_point=name, tests=tests, max_new_tokens=384)
    tools = [{'type': 'function', 'function': {'name': 'add', 'description': 'Return the sum of two integers.',
               'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                              'required': ['a', 'b'], 'additionalProperties': False}}}]
    for i in range(8):
        a, b = 19 + i * 3, -7 - i
        add(f'tool_{i}', 'tool', f'必须调用 add 工具计算 {a} 加 {b}，收到工具结果后只输出整数结果，不要解释。', a + b,
            mode='tool', tools=tools, arguments={'a': a, 'b': b})
    assert len(cases) == 150 and len({c['id'] for c in cases}) == len(cases)
    return cases


def build_compact():
    """Fixed category/template coverage; selection never reads model outcomes."""
    cases = build()
    def spread(rows, count):
        return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]
    selected = [row for row in cases if row['category'] == 'core']
    for prefix in ('multiply_', 'expression_', 'divide_'):
        selected += spread([row for row in cases if row['id'].startswith(prefix)], 4)
    for category, count in (('string', 6), ('reading', 6), ('logic', 6), ('coding', 6), ('tool', 4)):
        selected += spread([row for row in cases if row['category'] == category], count)
    identifiers = {row['id'] for row in selected}
    result = [row for row in cases if row['id'] in identifiers]
    assert len(result) == 50 and len(identifiers) == 50
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(50, 150), default=50)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = args.output or ROOT / ('benchmarks/task-quality-50.json' if args.size == 50 else
                                'benchmarks/task-quality-expanded.json')
    path.write_text(json.dumps(build_compact() if args.size == 50 else build(), ensure_ascii=False, indent=2) + '\n')
