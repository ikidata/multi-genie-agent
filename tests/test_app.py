"""
Tests for apps/app.py (Flask/Dash app entry point) and apps/logger.py

Covers: SSE streaming endpoint, logger setup
"""
import os
import sys
import json
import queue
import pytest
from unittest.mock import MagicMock, patch
from flask import Flask, Response, stream_with_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))


# =============================================================================
# Logger tests
# =============================================================================
class TestLogger:
    def test_activate_logger_returns_logger(self):
        from logger import activate_logger
        logger = activate_logger()
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')

    def test_logger_has_handler(self):
        from logger import activate_logger
        logger = activate_logger()
        assert len(logger.handlers) > 0

    def test_logger_does_not_duplicate_handlers(self):
        from logger import activate_logger
        logger1 = activate_logger()
        handler_count1 = len(logger1.handlers)
        logger2 = activate_logger()
        handler_count2 = len(logger2.handlers)
        assert handler_count1 == handler_count2

    def test_logger_propagation_disabled(self):
        from logger import activate_logger
        logger = activate_logger()
        assert logger.propagate is False


# =============================================================================
# SSE Endpoint tests (using standalone Flask app to test the SSE pattern)
# =============================================================================
class TestSSEEndpoint:
    """Test the SSE streaming pattern used in app.py."""

    def _make_flask_app(self):
        """Create a minimal Flask app with the SSE endpoint pattern."""
        app = Flask(__name__)
        app.chunk_queues = {}

        @app.route('/api/stream/<username_key>')
        def stream_events(username_key):
            def generate():
                chunk_queue = app.chunk_queues.get(username_key)
                if not chunk_queue:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'No active stream'})}\n\n"
                    return
                while True:
                    try:
                        chunk = chunk_queue.get(timeout=2)
                        if chunk == "__STREAMING_COMPLETE__":
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break
                        elif chunk.startswith("ERROR:"):
                            yield f"data: {json.dumps({'type': 'error', 'content': chunk})}\n\n"
                            break
                        else:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
                        break

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive'
                }
            )

        return app

    def test_sse_no_active_stream(self):
        app = self._make_flask_app()
        client = app.test_client()

        response = client.get('/api/stream/nonexistent_user')
        assert response.status_code == 200
        data = response.data.decode()
        assert 'No active stream' in data

    def test_sse_streams_chunks(self):
        app = self._make_flask_app()
        client = app.test_client()

        q = queue.Queue()
        q.put("Hello ")
        q.put("world!")
        q.put("__STREAMING_COMPLETE__")
        app.chunk_queues['testuser'] = q

        response = client.get('/api/stream/testuser')
        assert response.status_code == 200
        data = response.data.decode()
        assert 'Hello' in data
        assert 'world!' in data
        assert '"type": "done"' in data

    def test_sse_content_type(self):
        app = self._make_flask_app()
        client = app.test_client()

        q = queue.Queue()
        q.put("__STREAMING_COMPLETE__")
        app.chunk_queues['testuser'] = q

        response = client.get('/api/stream/testuser')
        assert 'text/event-stream' in response.content_type

    def test_sse_error_chunk(self):
        app = self._make_flask_app()
        client = app.test_client()

        q = queue.Queue()
        q.put("ERROR: Something went wrong")
        app.chunk_queues['erroruser'] = q

        response = client.get('/api/stream/erroruser')
        data = response.data.decode()
        assert '"type": "error"' in data
        assert 'Something went wrong' in data

    def test_sse_timeout(self):
        app = self._make_flask_app()
        client = app.test_client()

        # Put an empty queue - will timeout
        app.chunk_queues['timeoutuser'] = queue.Queue()

        response = client.get('/api/stream/timeoutuser')
        data = response.data.decode()
        assert '"type": "timeout"' in data

    def test_sse_cache_headers(self):
        app = self._make_flask_app()
        client = app.test_client()

        q = queue.Queue()
        q.put("__STREAMING_COMPLETE__")
        app.chunk_queues['testuser'] = q

        response = client.get('/api/stream/testuser')
        assert response.headers.get('Cache-Control') == 'no-cache'

    def test_sse_multiple_chunks(self):
        app = self._make_flask_app()
        client = app.test_client()

        q = queue.Queue()
        for i in range(10):
            q.put(f"chunk_{i} ")
        q.put("__STREAMING_COMPLETE__")
        app.chunk_queues['multiuser'] = q

        response = client.get('/api/stream/multiuser')
        data = response.data.decode()
        for i in range(10):
            assert f'chunk_{i}' in data
        assert '"type": "done"' in data
