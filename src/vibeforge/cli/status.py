"""Status includes persisted iteration decisions and actual committed refs."""


def status(database, campaign_id):
    return {**database.get(campaign_id), "iterations": database.iterations(campaign_id)}
