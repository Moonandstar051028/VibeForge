from vibeforge.controller.supervisor import Supervisor


def test_two_rounds_accept_good_change_reject_wrong_change_and_resume(paths, toy_config):
    controller = Supervisor(paths, "closed-loop")
    try:
        controller.create(toy_config)
        controller.run()
        state = controller.database.get("closed-loop")
        iterations = controller.database.iterations("closed-loop")
        assert state["state"] == "STOPPED"
        assert state["analysis_pending"] is False
        assert iterations[0]["decision"]["outcome"] == "accept"
        assert iterations[1]["decision"]["reason"] == "correctness_failed"
        assert state["stable_ref"] == iterations[0]["candidate"]
        assert state["stable_ref"] != iterations[1]["candidate"]
        assert state["previous_stable_ref"] == state["initial_ref"]
        controller.run(resume=True)
        assert controller.database.get("closed-loop")["completed_iterations"] == 2
        events = list(controller.database.events("closed-loop"))
        sessions = [e for e in events if e["event_type"] == "agent_completed"]
        assert len({e["payload"]["session_id"] for e in sessions}) == 3
        assert (paths.campaign("closed-loop") / "summary.md").exists()
    finally:
        controller.close()
