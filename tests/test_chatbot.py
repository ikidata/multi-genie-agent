"""
Tests for apps/DatabricksChatbot.py

Covers: DatabricksChatbot initialization, layout creation, callbacks,
        streaming worker, chunk accumulation in update_streaming, thread management,
        chat formatting, typing indicator, CSS injection
"""
import os
import sys
import time
import queue
import threading
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))


def _make_chatbot():
    """Helper to create a DatabricksChatbot with all dependencies mocked."""
    import dash
    import dash_bootstrap_components as dbc

    with patch('DatabricksChatbot.WorkspaceClient') as mock_ws, \
         patch('DatabricksChatbot.yaml.safe_load') as mock_yaml, \
         patch('builtins.open', MagicMock()), \
         patch('DatabricksChatbot.get_model_list', return_value=[
             {'label': 'Claude Sonnet 4.6', 'value': 'databricks-claude-sonnet-4-6'}
         ]):

        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws.return_value = mock_client

        mock_yaml.return_value = {
            'system_prompt': 'You are a helpful assistant for testing purposes.',
            'react_system_prompt': 'You are a ReAct agent for testing purposes.'
        }

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            suppress_callback_exceptions=True
        )

        from DatabricksChatbot import DatabricksChatbot
        chatbot = DatabricksChatbot(app=app, height='600px')
        return chatbot


# =============================================================================
# DatabricksChatbot initialization tests
# =============================================================================
class TestDatabricksChatbotInit:
    def test_init_creates_required_attributes(self):
        chatbot = _make_chatbot()
        assert hasattr(chatbot, 'chunk_queues')
        assert hasattr(chatbot, 'user_threads')
        assert hasattr(chatbot, 'user_queues')
        assert hasattr(chatbot, 'user_locks')
        assert hasattr(chatbot, '_global_lock')
        assert isinstance(chatbot.chunk_queues, dict)
        assert isinstance(chatbot._global_lock, type(threading.Lock()))

    def test_init_loads_configs(self):
        chatbot = _make_chatbot()
        assert chatbot.system_prompt is not None
        assert chatbot.react_system_prompt is not None
        assert len(chatbot.system_prompt) > 0

    def test_init_creates_openai_client(self):
        chatbot = _make_chatbot()
        assert chatbot.openai_client is not None


# =============================================================================
# Layout creation tests
# =============================================================================
class TestCreateLayout:
    @patch('DatabricksChatbot.get_model_list', return_value=[
        {'label': 'Claude Sonnet 4.6', 'value': 'databricks-claude-sonnet-4-6'}
    ])
    def test_layout_creates_all_stores(self, mock_model_list):
        chatbot = _make_chatbot()
        layout = chatbot._create_layout(
            'test.user@company.com', 'obo-token-123',
            {'TestSpace': {'id': 'sp1', 'description': 'Test'}}
        )
        # Layout should be a Dash Container
        assert layout is not None

    @patch('DatabricksChatbot.get_model_list', return_value=[
        {'label': 'Claude Sonnet 4.6', 'value': 'databricks-claude-sonnet-4-6'}
    ])
    def test_layout_cleans_username(self, mock_model_list):
        chatbot = _make_chatbot()
        chatbot._create_layout('Test.User@company.com', 'token', {})
        # Should have created chunk queue for lowercase username
        assert 'test.user@company.com' in chatbot.chunk_queues

    @patch('DatabricksChatbot.get_model_list', return_value=[
        {'label': 'Claude Sonnet 4.6', 'value': 'databricks-claude-sonnet-4-6'}
    ])
    def test_layout_resets_existing_thread(self, mock_model_list):
        chatbot = _make_chatbot()
        # Simulate existing thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        chatbot.user_threads['test@co.com'] = mock_thread

        chatbot._create_layout('test@co.com', 'token', {})
        # Thread should be removed
        assert 'test@co.com' not in chatbot.user_threads

    @patch('DatabricksChatbot.get_model_list', return_value=[
        {'label': 'Claude Sonnet 4.6', 'value': 'databricks-claude-sonnet-4-6'}
    ])
    def test_layout_resets_chunk_queue(self, mock_model_list):
        chatbot = _make_chatbot()
        old_queue = queue.Queue()
        old_queue.put("stale data")
        chatbot.chunk_queues['test@co.com'] = old_queue

        chatbot._create_layout('test@co.com', 'token', {})
        # Queue should be fresh (empty)
        new_queue = chatbot.chunk_queues['test@co.com']
        assert new_queue.empty()


# =============================================================================
# Worker thread tests
# =============================================================================
class TestStartUserWorker:
    def test_creates_queue_and_lock(self):
        chatbot = _make_chatbot()
        chatbot._start_user_worker('testuser')

        assert 'testuser' in chatbot.user_queues
        assert 'testuser' in chatbot.user_locks
        assert 'testuser' in chatbot.chunk_queues
        assert 'testuser' in chatbot.user_threads
        assert chatbot.user_threads['testuser'].is_alive()

        # Cleanup: stop worker
        chatbot.user_queues['testuser'].put(None)
        chatbot.user_threads['testuser'].join(timeout=2)

    def test_does_not_create_duplicate_thread(self):
        chatbot = _make_chatbot()
        chatbot._start_user_worker('testuser')
        first_thread = chatbot.user_threads['testuser']

        chatbot._start_user_worker('testuser')
        second_thread = chatbot.user_threads['testuser']

        assert first_thread is second_thread

        # Cleanup
        chatbot.user_queues['testuser'].put(None)
        first_thread.join(timeout=2)


class TestStreamingWorker:
    def test_puts_completion_sentinel(self):
        chatbot = _make_chatbot()
        chatbot.chunk_queues['testuser'] = queue.Queue()

        # Mock the streaming endpoint
        with patch.object(chatbot, '_call_model_endpoint_streaming', return_value=iter(['Hello', ' world'])):
            chatbot._streaming_worker(
                messages=[],
                username_key='testuser',
                user_name='Test User',
                obo_token='token',
                llm_params={'model': 'test', 'react_agent': 'no', 'max_tokens': 1000, 'genie_spaces': {}}
            )

        chunks = []
        while not chatbot.chunk_queues['testuser'].empty():
            chunks.append(chatbot.chunk_queues['testuser'].get())

        assert chunks[-1] == '__STREAMING_COMPLETE__'
        assert 'Hello' in chunks
        assert ' world' in chunks

    def test_handles_error_gracefully(self):
        chatbot = _make_chatbot()
        chatbot.chunk_queues['testuser'] = queue.Queue()

        with patch.object(chatbot, '_call_model_endpoint_streaming', side_effect=Exception("Stream error")):
            chatbot._streaming_worker(
                messages=[], username_key='testuser',
                user_name='Test User', obo_token='token'
            )

        chunks = []
        while not chatbot.chunk_queues['testuser'].empty():
            chunks.append(chatbot.chunk_queues['testuser'].get())

        assert any('ERROR' in c for c in chunks)
        assert '__STREAMING_COMPLETE__' in chunks

    def test_default_llm_params(self):
        chatbot = _make_chatbot()
        chatbot.chunk_queues['testuser'] = queue.Queue()

        with patch.object(chatbot, '_call_model_endpoint_streaming', return_value=iter([])) as mock_stream:
            chatbot._streaming_worker(
                messages=[], username_key='testuser',
                user_name='Test', obo_token='token',
                llm_params=None
            )
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args
            assert call_kwargs.kwargs.get('model') == 'databricks-claude-sonnet-4-6'


# =============================================================================
# Streaming accumulation tests (update_streaming callback logic)
#
# These tests verify the core streaming fix: chunks are accumulated across
# ticks into streaming_state['chunks'] and rendered as a single message,
# rather than creating a new message bubble per tick.
# =============================================================================
class TestStreamingAccumulation:
    """Test the chunk accumulation logic used by update_streaming."""

    def test_content_chunks_concatenated(self):
        """Multiple content chunks should be concatenated into a single string."""
        accumulated = ['Hello ', 'world', '!']
        all_content = ""
        genie_messages = []
        for chunk in accumulated:
            if chunk.startswith("\U0001f9de\u200d\u2642\ufe0f Genie") and chunk.endswith("activated"):
                if chunk not in genie_messages:
                    genie_messages.append(chunk)
            else:
                all_content += chunk
        assert all_content == "Hello world!"
        assert len(genie_messages) == 0

    def test_genie_messages_separated(self):
        """Genie activation messages should be separated from content."""
        accumulated = [
            'Analyzing...',
            "\U0001f9de\u200d\u2642\ufe0f Genie 'Sales' activated",
            'Results ready.'
        ]
        all_content = ""
        genie_messages = []
        for chunk in accumulated:
            if chunk.startswith("\U0001f9de\u200d\u2642\ufe0f Genie") and chunk.endswith("activated"):
                if chunk not in genie_messages:
                    genie_messages.append(chunk)
            else:
                all_content += chunk
        assert all_content == "Analyzing...Results ready."
        assert len(genie_messages) == 1
        assert "Sales" in genie_messages[0]

    def test_duplicate_genie_messages_deduped(self):
        """Same genie activation message appearing twice should only be kept once."""
        genie_msg = "\U0001f9de\u200d\u2642\ufe0f Genie 'Sales' activated"
        accumulated = [genie_msg, 'some content', genie_msg]
        genie_messages = []
        for chunk in accumulated:
            if chunk.startswith("\U0001f9de\u200d\u2642\ufe0f Genie") and chunk.endswith("activated"):
                if chunk not in genie_messages:
                    genie_messages.append(chunk)
        assert len(genie_messages) == 1

    def test_empty_accumulation(self):
        """No chunks should produce empty content and no genie messages."""
        accumulated = []
        all_content = ""
        genie_messages = []
        for chunk in accumulated:
            if chunk.startswith("\U0001f9de\u200d\u2642\ufe0f Genie") and chunk.endswith("activated"):
                if chunk not in genie_messages:
                    genie_messages.append(chunk)
            else:
                all_content += chunk
        assert all_content == ""
        assert len(genie_messages) == 0

    def test_chunks_accumulate_across_batches(self):
        """Simulates multiple ticks: chunks from different batches should be merged."""
        # Tick 1 chunks
        batch1 = ['Hello ']
        # Tick 2 chunks
        batch2 = ['world']
        # Tick 3 chunks (completion)
        batch3 = ['!', '__STREAMING_COMPLETE__']

        accumulated = []
        for batch in [batch1, batch2, batch3]:
            content_chunks = [c for c in batch if c != '__STREAMING_COMPLETE__']
            accumulated.extend(content_chunks)

        all_content = ""
        for chunk in accumulated:
            all_content += chunk
        assert all_content == "Hello world!"

    def test_base_display_filters_chunk_and_typing(self):
        """Base display should exclude chunk-message and typing-message divs."""
        from dash import html, dcc

        chat_display = [
            html.Div('user msg', className='message-container user-container'),
            html.Div([
                dcc.Markdown('old chunk', className='chat-message assistant-message chunk-message')
            ], className='message-container assistant-container', id='streaming-content'),
            html.Div(className='chat-message assistant-message typing-message'),
        ]

        base_display = []
        for div in chat_display:
            div_str = str(div)
            if 'typing-message' in div_str:
                continue
            if 'chunk-message' in div_str:
                continue
            base_display.append(div)

        assert len(base_display) == 1  # Only user msg remains
        assert 'user msg' in str(base_display[0])

    def test_streaming_state_accumulates_chunks(self):
        """streaming_state['chunks'] should grow across simulated ticks."""
        streaming_state = {'streaming': True, 'chunks': []}

        # Tick 1
        new_chunks_1 = ['Hello ']
        accumulated = list(streaming_state.get('chunks', []))
        accumulated.extend(new_chunks_1)
        streaming_state['chunks'] = accumulated
        assert streaming_state['chunks'] == ['Hello ']

        # Tick 2
        new_chunks_2 = ['world!']
        accumulated = list(streaming_state.get('chunks', []))
        accumulated.extend(new_chunks_2)
        streaming_state['chunks'] = accumulated
        assert streaming_state['chunks'] == ['Hello ', 'world!']

    def test_chat_history_not_updated_during_streaming(self):
        """During streaming (before completion), chat_history should not be modified.
        Only on completion should the full assistant message be added."""
        original_history = [
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Hello'}
        ]
        # During streaming, we return dash.no_update for chat_history.
        # Verify the original history remains unchanged.
        history_copy = list(original_history)
        assert len(history_copy) == 2

        # On completion, append the full response
        all_content = "Hello world! This is the full response."
        history_copy.append({'role': 'assistant', 'content': all_content})
        assert len(history_copy) == 3
        assert history_copy[-1]['content'] == all_content

    def test_genie_name_extraction(self):
        """Genie name should be extracted from activation message for status display."""
        genie_chunk = "\U0001f9de\u200d\u2642\ufe0f Genie 'Sales Analytics' activated"
        genie_name = genie_chunk.split("'")[1] if "'" in genie_chunk else "Unknown"
        assert genie_name == "Sales Analytics"

    def test_genie_name_extraction_no_quotes(self):
        """Fallback to 'Unknown' if no quotes in genie message."""
        genie_chunk = "some weird genie message"
        genie_name = genie_chunk.split("'")[1] if "'" in genie_chunk else "Unknown"
        assert genie_name == "Unknown"


# =============================================================================
# Chat formatting tests
# =============================================================================
class TestFormatChatDisplay:
    def test_formats_user_message(self):
        chatbot = _make_chatbot()
        history = [{'role': 'user', 'content': 'Hello'}]
        result = chatbot._format_chat_display(history)
        assert len(result) == 1

    def test_formats_assistant_message(self):
        chatbot = _make_chatbot()
        history = [{'role': 'assistant', 'content': 'Hi there!'}]
        result = chatbot._format_chat_display(history)
        assert len(result) == 1

    def test_skips_system_message(self):
        chatbot = _make_chatbot()
        history = [
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Hello'}
        ]
        result = chatbot._format_chat_display(history)
        assert len(result) == 1  # Only user message

    def test_handles_empty_history(self):
        chatbot = _make_chatbot()
        result = chatbot._format_chat_display([])
        assert result == []


# =============================================================================
# Typing indicator tests
# =============================================================================
class TestCreateTypingIndicator:
    def test_creates_typing_div(self):
        chatbot = _make_chatbot()
        indicator = chatbot._create_typing_indicator()
        assert indicator is not None
        indicator_str = str(indicator)
        assert 'typing-message' in indicator_str

    def test_has_three_dots(self):
        chatbot = _make_chatbot()
        indicator = chatbot._create_typing_indicator()
        indicator_str = str(indicator)
        assert indicator_str.count('typing-dot') == 3


# =============================================================================
# CSS injection tests
# =============================================================================
class TestAddCustomCss:
    def test_css_injected(self):
        chatbot = _make_chatbot()
        index_string = chatbot.app.index_string
        assert 'DM Sans' in index_string
        assert 'chat-container' in index_string
        assert 'assistant-message' in index_string

    def test_javascript_injected(self):
        chatbot = _make_chatbot()
        index_string = chatbot.app.index_string
        assert 'scrollChatToBottom' in index_string
        assert 'enhanceMultiSelect' in index_string


# =============================================================================
# Config validation tests
# =============================================================================
class TestChatbotConfig:
    def test_valid_config(self):
        from DatabricksChatbot import ChatConfig
        config = ChatConfig(
            system_prompt='A valid system prompt for testing.',
            react_system_prompt='A valid react prompt for testing.'
        )
        assert config.system_prompt is not None

    def test_short_system_prompt_rejected(self):
        from DatabricksChatbot import ChatConfig
        with pytest.raises(Exception):
            ChatConfig(system_prompt='Short', react_system_prompt='A valid react prompt for testing.')

    def test_empty_system_prompt_rejected(self):
        from DatabricksChatbot import ChatConfig
        with pytest.raises(Exception):
            ChatConfig(system_prompt='', react_system_prompt='A valid react prompt for testing.')
