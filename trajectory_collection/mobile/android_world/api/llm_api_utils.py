"""Thin client layer for OpenAI-compatible chat-completion endpoints.

Used by the collection agents in android_world.agents.infer. Provides:
- A thread-local httpx client, because httpx.Client is not thread-safe and a
  shared client corrupts its connection pool under ThreadPoolExecutor.
- Optional JSONL logging of API calls (latency + token usage, no payloads).
"""

import json
import os
import threading
import time

import httpx
from openai import OpenAI

_api_logger_enabled = True
_api_log_dir = './api_logs'

_thread_local = threading.local()


def _get_thread_local_httpx_client(timeout: httpx.Timeout) -> httpx.Client:
  """Returns this thread's httpx client, creating it on first use."""
  if getattr(_thread_local, 'httpx_client', None) is None:
    _thread_local.httpx_client = httpx.Client(
        timeout=timeout,
        limits=httpx.Limits(
            max_keepalive_connections=10,
            max_connections=25,
            keepalive_expiry=300.0,
        ),
    )
  return _thread_local.httpx_client


def cleanup_thread_local_httpx_client() -> None:
  """Closes this thread's httpx client, if any."""
  client = getattr(_thread_local, 'httpx_client', None)
  if client is not None:
    try:
      client.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    finally:
      _thread_local.httpx_client = None


def initialize_api_logger(enable: bool = True, log_dir: str = './api_logs'):
  """Configures API-call logging. Call once at startup (optional)."""
  global _api_logger_enabled, _api_log_dir
  _api_logger_enabled = enable
  _api_log_dir = log_dir


def _log_api_call(
    model_name: str,
    duration_ms: float,
    usage: dict | None,
    success: bool = True,
    error: str | None = None,
) -> None:
  """Appends one JSONL record per API call. Never raises."""
  if not _api_logger_enabled:
    return
  try:
    os.makedirs(_api_log_dir, exist_ok=True)
    record = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'model': model_name,
        'duration_ms': round(duration_ms, 1),
        'usage': usage,
        'success': success,
        'error': error,
    }
    path = os.path.join(_api_log_dir, 'api_calls.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(record, ensure_ascii=False) + '\n')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'Warning: failed to log API call: {e}')


def get_initial_messages(user_msg_content, system_msg_content=None):
  """Builds the initial messages list, with non-empty content guaranteed."""
  user_content = user_msg_content
  if not user_content or not user_content.strip():
    user_content = 'Please help me with this task.'
  if system_msg_content and system_msg_content.strip():
    return [
        {'role': 'system', 'content': system_msg_content},
        {'role': 'user', 'content': user_content},
    ]
  return [{'role': 'user', 'content': user_content}]


def get_model_response(
    model_url,
    model_name,
    model_token,
    messages,
    tool_schemas=None,
    agent_logger=None,
    max_retry_num=10,
    temperature=None,
    max_tokens=None,
    timeout=180,
):
  """Calls an OpenAI-compatible chat-completion endpoint with retries.

  Args:
    model_url: Base URL of the endpoint (should end with /v1).
    model_name: Model identifier.
    model_token: API key.
    messages: Chat messages in OpenAI format.
    tool_schemas: Optional tool schemas passed through to the API.
    agent_logger: Optional logging.Logger for debug/error output.
    max_retry_num: Maximum attempts before giving up.
    temperature: Optional sampling temperature.
    max_tokens: Optional completion token cap.
    timeout: Read timeout in seconds.

  Returns:
    The full ChatCompletion response object.

  Raises:
    ValueError: If all attempts fail.
  """
  call_start_time = time.time()
  httpx_timeout = httpx.Timeout(
      connect=30.0, read=timeout, write=60.0, pool=30.0
  )

  for retry_num in range(max_retry_num):
    try:
      client = OpenAI(
          api_key=model_token,
          base_url=model_url,
          http_client=_get_thread_local_httpx_client(httpx_timeout),
      )
      request_params = {'model': model_name, 'messages': messages}
      if temperature is not None:
        request_params['temperature'] = temperature
      if max_tokens is not None:
        request_params['max_tokens'] = max_tokens
      if tool_schemas is not None:
        request_params['tools'] = tool_schemas

      response = client.chat.completions.create(**request_params)

      usage = None
      if response.usage:
        usage = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
      _log_api_call(
          model_name=model_name,
          duration_ms=(time.time() - call_start_time) * 1000,
          usage=usage,
          success=True,
      )
      return response
    except Exception as e:  # pylint: disable=broad-exception-caught
      if agent_logger is not None:
        agent_logger.error(
            'API request failed (attempt %d/%d): %s',
            retry_num + 1,
            max_retry_num,
            e,
        )
      if retry_num < max_retry_num - 1:
        backoff_delay = min(2**retry_num, 60)
        time.sleep(backoff_delay)
      else:
        _log_api_call(
            model_name=model_name,
            duration_ms=(time.time() - call_start_time) * 1000,
            usage=None,
            success=False,
            error=str(e),
        )
        raise ValueError(
            f'API request failed after {max_retry_num} attempts: {e}'
        ) from e
