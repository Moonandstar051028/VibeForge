"""Command line interface for profiles, managed inference, and campaigns."""

import argparse
import json
import traceback
from pathlib import Path
from vibeforge.agent.runner import worker
from vibeforge.cli.campaign import launch
from vibeforge.cli.status import status
from vibeforge.contracts.campaign import load_config
from vibeforge.engine.server import serve
from vibeforge.profiling.collector import collect
from vibeforge.storage.atomic import write_json
from vibeforge.storage.paths import Paths
from vibeforge.storage.database import Database
from vibeforge.observability.reports import write_campaign_report


def parser():
    result = argparse.ArgumentParser(prog="vibeforge")
    result.add_argument("--project", default=str(Path.cwd()))
    result.add_argument("--data-root")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("profile", "run", "resume", "status", "stop", "events", "report"):
        command = commands.add_parser(name)
        command.add_argument("campaign", nargs="?")
        if name in ("profile", "run"):
            command.add_argument("--config", default="configs/campaigns/qwen36-35b.json")
        if name == "events":
            command.add_argument("--after", type=int, default=0)
    command = commands.add_parser("serve")
    command.add_argument("--directory", required=True)
    command.add_argument("--port", type=int, default=0)
    command.add_argument("--token", required=True)
    command = commands.add_parser("agent-worker")
    for name in ("context", "workspace", "trace", "result", "url", "prompt", "token"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--analysis-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    project = Path(args.project).resolve()
    if args.command == "serve":
        serve(args.directory, args.port, args.token)
        return 0
    if args.command == "agent-worker":
        worker(args.context, args.workspace, args.trace, args.result, args.url,
               args.prompt, args.token, args.analysis_only)
        return 0
    config = None
    campaign_id = args.campaign
    if args.command in ("profile", "run"):
        config = load_config(project / args.config, project)
        campaign_id = campaign_id or config.get("name", Path(args.config).stem)
    elif not campaign_id:
        raise ValueError(f"{args.command} requires a campaign ID")
    paths = Paths.create(project, args.data_root or (config.get("data_root") if config else None))
    if args.command == "profile":
        profile, preflight = collect(config)
        write_json(paths.data / "profiles" / f"{profile['profile_hash']}.json", profile)
        print(json.dumps({"profile": profile, "preflight": preflight}, indent=2, ensure_ascii=False))
        return 0 if preflight["passed"] else 2
    if args.command in ("run", "resume"):
        state = launch(paths, campaign_id, config, resume=args.command == "resume")
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0 if state["state"] == "STOPPED" else 2
    database = Database(paths.data / "state" / "controller.sqlite3")
    try:
        if args.command == "status":
            print(json.dumps(status(database, campaign_id), indent=2, ensure_ascii=False))
        elif args.command == "events":
            for event in database.events(campaign_id, args.after):
                print(json.dumps(event, ensure_ascii=False))
        elif args.command == "report":
            print(write_campaign_report(database, paths, campaign_id))
        elif args.command == "stop":
            database.update(campaign_id, {"stop_requested": True}, "stop_requested")
            print(json.dumps({"campaign_id": campaign_id, "stop_requested": True}))
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
