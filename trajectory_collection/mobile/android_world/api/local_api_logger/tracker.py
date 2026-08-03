"""
API call tracker.
Provides decorators and wrappers to log API calls automatically.
"""

import time
import json
from typing import Dict, Any, Optional, Callable
from functools import wraps
from .logger import APILogger, _default_logger


class APITracker:
    """API call tracker."""

    def __init__(self, logger: Optional[APILogger] = None):
        """
        Initialize the tracker.

        Args:
            logger: Logger instance, defaults to the global instance.
        """
        self.logger = logger or _default_logger

    def track_request(
        self,
        model: str,
        user: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator: automatically track a function call and log it.

        Usage:
            @tracker.track_request(model="claude-3-opus", user="john")
            def call_api():
                # ... API call code
                return response

        Args:
            model: Model name.
            user: User identifier.
            metadata: Extra metadata.
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                error = None
                success = True
                response_data = {}

                try:
                    result = func(*args, **kwargs)
                    response_data = result if isinstance(result, dict) else {"result": str(result)}
                    return result
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000

                    # Extract request data from kwargs or args
                    request_data = self._extract_request_data(args, kwargs)

                    self.logger.log_call(
                        model=model,
                        request_data=request_data,
                        response_data=response_data,
                        user=user,
                        duration_ms=duration_ms,
                        success=success,
                        error=error,
                        metadata=metadata
                    )

            return wrapper
        return decorator

    def log_completion(
        self,
        model: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        api_key: Optional[str] = None,
        user: str = "default",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Manually log one API call.

        Args:
            model: Model name.
            request_data: Request data.
            response_data: Response data (must contain the provider-returned usage field).
            user: User identifier.
            duration_ms: Call duration.
            metadata: Extra metadata.
        """
        self.logger.log_call(
            model=model,
            request_data=request_data,
            response_data=response_data,
            user=user,
            duration_ms=duration_ms,
            success=True,
            metadata=metadata,
            api_key=api_key
        )

    def wrap_requests_call(
        self,
        model: str,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        user: str = "default",
        verify: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Wrap a requests POST call and log it automatically.

        Args:
            model: Model name.
            url: API endpoint.
            headers: Request headers.
            payload: Request data.
            user: User identifier.
            verify: SSL verification.
            timeout: Timeout in seconds.

        Returns:
            API response data.
        """
        import requests

        start_time = time.time()

        try:
            # For streaming requests, add stream_options automatically
            if payload.get("stream", False):
                if "stream_options" not in payload:
                    payload["stream_options"] = {"include_usage": True}

                # Handle streaming response
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    verify=verify,
                    timeout=timeout,
                    stream=True
                )
                response.raise_for_status()

                return self._handle_stream_response(
                    response, model, payload, user, start_time
                )
            else:
                # Handle non-streaming response
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    verify=verify,
                    timeout=timeout
                )
                response.raise_for_status()

                duration_ms = (time.time() - start_time) * 1000
                response_data = response.json()

                self.logger.log_call(
                    model=model,
                    request_data=payload,
                    response_data=response_data,
                    user=user,
                    duration_ms=duration_ms,
                    success=True
                )

                return response_data

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log the error
            self.logger.log_call(
                model=model,
                request_data=payload,
                response_data={},
                user=user,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )

            raise

    def _handle_stream_response(
        self,
        response,
        model: str,
        request_data: Dict[str, Any],
        user: str,
        start_time: float
    ):
        """
        Handle a streaming response and log it.

        Returns a generator that yields response chunks one by one.
        """
        collected_content = []
        collected_usage = None

        def stream_generator():
            nonlocal collected_usage

            try:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')

                        # Pass through to the caller unchanged
                        yield line_str + '\n'

                        # Parse and collect data
                        if line_str.startswith('data: '):
                            data_str = line_str[6:].strip()
                            if data_str != '[DONE]':
                                try:
                                    data = json.loads(data_str)

                                    # Collect content
                                    if "choices" in data:
                                        for choice in data["choices"]:
                                            if "delta" in choice and "content" in choice["delta"]:
                                                content = choice["delta"].get("content")
                                                if content:
                                                    collected_content.append(content)

                                    # Collect usage info
                                    if "usage" in data and data["usage"] is not None:
                                        collected_usage = data["usage"]

                                except json.JSONDecodeError:
                                    pass
            finally:
                # Log after the stream ends
                duration_ms = (time.time() - start_time) * 1000
                completion_content = "".join(collected_content)

                response_data = {
                    "content": completion_content,
                    "streaming": True
                }

                # Add usage info if available
                if collected_usage:
                    response_data["usage"] = collected_usage

                self.logger.log_call(
                    model=model,
                    request_data=request_data,
                    response_data=response_data,
                    user=user,
                    duration_ms=duration_ms,
                    success=True
                )

        return stream_generator()

    def _extract_request_data(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract request data from function arguments."""
        request_data = {}

        # Extract common fields from kwargs
        for key in ["messages", "prompt", "model", "temperature", "max_tokens"]:
            if key in kwargs:
                request_data[key] = kwargs[key]

        # If the first positional arg is a dict, treat it as request data
        if args and isinstance(args[0], dict):
            request_data.update(args[0])

        return request_data


# Global default tracker
_default_tracker = APITracker()


def track_request(model: str, user: str = "default", metadata: Optional[Dict[str, Any]] = None):
    """Decorator using the default tracker."""
    return _default_tracker.track_request(model, user, metadata)


def log_completion(
    model: str,
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    api_key: Optional[str] = None,
    user: str = "default",
    duration_ms: Optional[float] = None
):
    """Log a call using the default tracker."""
    return _default_tracker.log_completion(
        model=model, 
        request_data=request_data, 
        response_data=response_data, 
        api_key=api_key, 
        user=user, 
        duration_ms=duration_ms
    )


def wrap_requests_call(
    model: str,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    user: str = "default",
    **kwargs
):
    """Wrap a requests call using the default tracker."""
    return _default_tracker.wrap_requests_call(model, url, headers, payload, user, **kwargs)
