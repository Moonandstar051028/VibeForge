"""Apply the backend's Landlock write policy before executing an Agent command."""

import json
import os
import sys

from .sandbox import restrict_writes


def command_argv(command, writable):
    # A separate interpreter avoids preexec_fn in the threaded Agent process.
    return [sys.executable, '-m', __name__, json.dumps([str(p) for p in writable]), *command]


def main():
    writable, command = json.loads(sys.argv[1]), sys.argv[2:]
    if not command:
        raise ValueError('Missing sandbox command')
    restrict_writes(writable)
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
