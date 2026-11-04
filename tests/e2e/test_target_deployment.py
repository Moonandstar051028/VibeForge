import os
import uuid
import pytest
from vibeforge.contracts.campaign import load_config
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import read_json


@pytest.mark.target
@pytest.mark.skipif(os.environ.get("VIBEFORGE_TARGET_TEST") != "1",
                    reason="Enable VIBEFORGE_TARGET_TEST=1 to attempt the real local model")
def test_real_qwen_deployment_and_agent(paths, project):
    config = load_config(project / "configs/campaigns/qwen36-35b.json", project)
    controller = Supervisor(paths, "target-" + uuid.uuid4().hex[:8])
    try:
        controller.create(config)
        state = controller.run()
        assert state["state"] == "STOPPED", (
            f"Real model deployment did not complete: {state['state']}: {state.get('stop_reason')}; "
            f"evidence: {controller.directory}")
        assert state["mode"] == "local_model"
        assert state["stable_ref"]
        assert read_json(controller.directory / 'bootstrap/agent-smoke.json')['passed']
        agent_results = [read_json(path) for path in
                         (controller.directory / 'iterations').glob('*/agent-result.json')]
        assert any(result.get('mode') == 'qwen_agent' and result.get('complete') for result in agent_results), (
            'The real model must complete at least one tool session, not only start its HTTP server')
    finally:
        controller.close()
