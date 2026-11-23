"""Exit races must not prevent feedback or authorize signalling unrelated jobs."""
from pathlib import Path
from unittest.mock import patch

import pytest

from vf_runtime.engine.process import stop_record
from vf_runtime.storage.atomic import write_json


def test_exit_between_identity_and_cmdline_is_cleaned(tmp_path):
    record = tmp_path / 'engine.json'
    write_json(record, {'pid': 123, 'token': 'owned'})
    with patch('vf_runtime.engine.process.same_process', side_effect=[True, False]), \
         patch.object(Path, 'read_bytes', return_value=b''), \
         patch('vf_runtime.engine.process.os.killpg') as kill:
        assert stop_record(record)
        kill.assert_not_called()
    assert not record.exists()


def test_live_unowned_process_is_never_signalled(tmp_path):
    record = tmp_path / 'engine.json'
    write_json(record, {'pid': 123, 'token': 'owned'})
    with patch('vf_runtime.engine.process.same_process', return_value=True), \
         patch.object(Path, 'read_bytes', return_value=b'unrelated-process'), \
         patch('vf_runtime.engine.process.os.killpg') as kill:
        with pytest.raises(RuntimeError, match='ownership token'):
            stop_record(record)
        kill.assert_not_called()
    assert record.exists()
