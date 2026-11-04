"""Campaign lifecycle entry points shared by CLI commands."""

from vibeforge.controller.supervisor import Supervisor


def launch(paths, campaign_id, config=None, resume=False):
    supervisor = Supervisor(paths, campaign_id)
    try:
        if config is not None:
            supervisor.create(config)
        supervisor.run(resume=resume)
        return supervisor.database.get(campaign_id)
    finally:
        supervisor.close()
