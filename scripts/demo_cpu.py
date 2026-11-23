"""Three real CPU controller rounds using explicitly scripted edits, not an LLM."""
import argparse
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from vf_runtime.storage.atomic import atomic_write, write_json
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.loop import Pipeline


def connect(url, config, workspace, folder):
    report = {'passed': True, 'fixture': True, 'note': 'Scripted CPU demonstration, NOT real agent readiness'}
    write_json(Path(folder) / 'connection.json', report)
    return report


def edit(url, config, workspace, folder, context):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / 'context.md', context)
    number = int(folder.parent.name)
    source = Path(workspace) / 'engine.py'
    text = source.read_text()
    if number == 1:
        text = text.replace('sum(range(120000))', '(120000 * 119999 // 2)')
        explanation = 'CPU fixture: replace a repeated arithmetic sum with its equivalent closed form.'
    elif number == 2:
        text = text.replace('/ 997 for index in indices]', '/ 997 + 0.1 for index in indices]')
        explanation = 'CPU fixture: deliberately perturb logits to test rejection and rollback.'
    else:
        text += '\ndef broken(:\n'
        explanation = 'CPU fixture: deliberately introduce a syntax error to test build failure.'
    source.write_text(text)
    atomic_write(folder.parent / 'analysis.md', explanation + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Pipeline(load(ROOT / 'configs/toy.json'), args.output, edit=edit, connect=connect).execute(rounds=3)
