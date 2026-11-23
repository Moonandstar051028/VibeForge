"""Small command line: run, resume, status, profile."""

import argparse
import json
import signal
from pathlib import Path

from vf_runtime.engine.client import EngineClient
from vf_runtime.storage.atomic import read_json, write_json
from .config import load
from .loop import Pipeline


def main():
    parser = argparse.ArgumentParser(description='VibeForge v2: self-hosted file-based optimization')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', required=True)
    run.add_argument('--output', required=True)
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument('--rounds', type=int)
    run_mode.add_argument('--continuous', action='store_true', default=None,
                          help='Keep iterating until stopped or an integrity/startup failure occurs')
    resume = commands.add_parser('resume')
    resume.add_argument('--output', required=True)
    resume_mode = resume.add_mutually_exclusive_group()
    resume_mode.add_argument('--rounds', type=int, help='Total round count including completed rounds')
    resume_mode.add_argument('--continuous', action='store_true', default=None,
                             help='Resume without a round limit; continuous mode is inherited by default')
    status = commands.add_parser('status')
    status.add_argument('--output', required=True)
    profile = commands.add_parser('profile')
    profile.add_argument('--url', required=True)
    profile.add_argument('--prompt', required=True)
    profile.add_argument('--tokens', type=int, default=16)
    profile.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(read_json(Path(args.output) / 'state.json'), ensure_ascii=False, indent=2))
    elif args.command == 'profile':
        result = EngineClient(args.url, 900).profile(prompt=args.prompt, max_new_tokens=args.tokens)
        write_json(args.output, result)
    else:
        if args.rounds is not None and args.rounds < 1:
            parser.error('--rounds must be positive')
        config = load(args.config) if args.command == 'run' else read_json(Path(args.output) / 'config.json')
        # Detached continuous runs must retain the normal interruption cleanup.
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in previous:
                signal.signal(sig, signal.default_int_handler)
            Pipeline(config, args.output).execute(args.rounds, resume=args.command == 'resume',
                                                 continuous=args.continuous)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    main()
