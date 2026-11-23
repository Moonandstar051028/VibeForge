"""Persist actual Agent adapter inputs and model responses without changing them."""
import copy
import time
from pathlib import Path

from vf_runtime.storage.atomic import write_json


def plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def attach_trace(llm, folder):
    """Capture before folding and after folding/native conversion; never call the LLM twice."""
    folder = Path(folder) / 'requests'
    original_chat = llm.chat
    original_create = llm._chat_complete_create
    pending = []
    sequence = 0

    def chat(*args, **kwargs):
        nonlocal pending
        pending = copy.deepcopy(plain(kwargs.get('messages', args[0] if args else [])))
        return original_chat(*args, **kwargs)

    def create(*args, **kwargs):
        nonlocal sequence
        sequence += 1
        target = folder / f'{sequence:05d}'
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / 'before-fold.json', pending)
        write_json(target / 'request.json', {
            'at': time.time(), 'scope': 'Actual native adapter arguments after Qwen-Agent folding and conversion. '
            'The SDK adapter subsequently moves top_k/repetition_penalty into extra_body and request_timeout into timeout; '
            'this is not a network packet capture.', 'arguments': plain(kwargs)})
        result = {'started_at': time.time(), 'status': 'running', 'chunks': [],
                  'reasoning_content': '', 'content': '', 'tool_calls': [], 'finish_reasons': []}
        write_json(target / 'response.json', result)

        def record(chunk):
            data = plain(chunk)
            result['chunks'].append(data)
            for choice in data.get('choices', []):
                delta = choice.get('delta') or choice.get('message') or {}
                result['reasoning_content'] += delta.get('reasoning_content') or ''
                result['content'] += delta.get('content') or ''
                result['tool_calls'].extend(delta.get('tool_calls') or [])
                if choice.get('finish_reason'):
                    result['finish_reasons'].append(choice['finish_reason'])
            write_json(target / 'response.json', result)

        def finish(status, error=None):
            result.update(status=status, finished_at=time.time())
            if error is not None:
                result['error'] = f'{type(error).__name__}: {error}'
            write_json(target / 'response.json', result)

        try:
            response = original_create(*args, **kwargs)
        except BaseException as error:
            finish('failed', error)
            raise
        if not kwargs.get('stream'):
            record(response)
            finish('completed')
            return response

        def stream():
            try:
                for chunk in response:
                    record(chunk)
                    yield chunk
                finish('completed')
            except BaseException as error:
                finish('interrupted' if not isinstance(error, Exception) else 'failed', error)
                raise
            finally:
                if hasattr(response, 'close'):
                    response.close()
        return stream()

    llm.chat = chat
    llm._chat_complete_create = create
