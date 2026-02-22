"""
Tests for apps/agent.py (ChatBotAgent) and apps/simple_agent.py (SimpleChatBotAgent)

Covers: agent initialization, stringify_tool_call, process_tool_calls,
        agent_tool_calling, predict, predict_stream
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))


# =============================================================================
# ChatBotAgent (ReAct) tests
# =============================================================================
class TestChatBotAgent:
    @patch('agent.WorkspaceClient')
    def test_init_with_genie_spaces(self, mock_ws_cls, sample_genie_spaces):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        agent = ChatBotAgent(
            model_name='databricks-claude-sonnet-4-6',
            max_tokens=1000,
            obo_token='test-token',
            genie_spaces=sample_genie_spaces
        )
        assert len(agent.tools) == 3
        assert agent.model_name == 'databricks-claude-sonnet-4-6'
        assert agent.max_tokens == 1000

    @patch('agent.WorkspaceClient')
    def test_init_without_genie_spaces(self, mock_ws_cls):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        agent = ChatBotAgent(
            model_name='databricks-claude-sonnet-4-6',
            max_tokens=500
        )
        assert agent.tools == []
        assert agent.obo_token == ''

    @patch('agent.WorkspaceClient')
    def test_stringify_tool_call(self, mock_ws_cls, mock_llm_tool_response):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        result = agent.stringify_tool_call(mock_llm_tool_response)
        assert result.role == 'assistant'
        assert result.tool_calls is not None

    @patch('agent.WorkspaceClient')
    def test_stringify_tool_call_with_content(self, mock_ws_cls):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        response = MagicMock()
        response.id = 'resp-1'
        response.choices = [MagicMock()]
        response.choices[0].message.role = 'assistant'
        response.choices[0].message.content = 'Thinking about this...'
        tc = MagicMock()
        tc.id = 'tc-1'
        tc.type = 'function'
        tc.function.name = 'genie_x'
        tc.function.arguments = '{"prompt": "q"}'
        response.choices[0].message.tool_calls = [tc]

        result = agent.stringify_tool_call(response)
        assert result.content == 'Thinking about this...'

    @patch('agent.WorkspaceClient')
    def test_stringify_tool_call_invalid_response(self, mock_ws_cls):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        bad_response = MagicMock()
        bad_response.choices = []

        with pytest.raises(ValueError, match="Invalid response format"):
            agent.stringify_tool_call(bad_response)

    @patch('agent.run_genie')
    @patch('agent.WorkspaceClient')
    def test_process_tool_calls_genie(self, mock_ws_cls, mock_run_genie, mock_llm_tool_response):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        mock_run_genie.return_value = ('Revenue is $1M', 'conv-123')

        from agent import ChatBotAgent
        agent = ChatBotAgent(
            model_name='test', max_tokens=1000,
            obo_token='test-token'
        )

        messages = agent.process_tool_calls(mock_llm_tool_response)
        assert len(messages) == 1
        assert messages[0].role == 'tool'
        assert 'Revenue is $1M' in messages[0].content

    @patch('agent.run_genie')
    @patch('agent.WorkspaceClient')
    def test_process_tool_calls_updates_conversation_dict(self, mock_ws_cls, mock_run_genie, mock_llm_tool_response):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        mock_run_genie.return_value = ('Result', 'conv-abc')

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000, obo_token='tok')

        agent.process_tool_calls(mock_llm_tool_response)
        assert 'genie_space123' in agent.genie_conversation_dict
        assert agent.genie_conversation_dict['genie_space123'] == 'conv-abc'

    @patch('agent.call_chat_model')
    @patch('agent.WorkspaceClient')
    def test_agent_tool_calling_stop_immediately(self, mock_ws_cls, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        # LLM returns stop immediately
        response = MagicMock()
        response.id = 'resp-1'
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = 'stop'
        response.choices[0].message.role = 'assistant'
        response.choices[0].message.content = 'Direct answer'
        response.choices[0].message.tool_calls = None
        response.choices[0].message.to_dict.return_value = {
            'role': 'assistant', 'content': 'Direct answer'
        }
        mock_call.return_value = response

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        results = list(agent.agent_tool_calling(messages=[]))
        assert len(results) == 1
        assert results[0].content == 'Direct answer'

    @patch('agent.call_chat_model')
    @patch('agent.WorkspaceClient')
    def test_agent_tool_calling_length_warning(self, mock_ws_cls, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        response = MagicMock()
        response.id = 'resp-1'
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = 'length'
        response.choices[0].message.tool_calls = None
        mock_call.return_value = response

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        results = list(agent.agent_tool_calling(messages=[]))
        assert any('running out of tokens' in str(r) for r in results)

    @patch('agent.call_chat_model')
    @patch('agent.WorkspaceClient')
    def test_agent_tool_calling_error_handling(self, mock_ws_cls, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        mock_call.side_effect = Exception("LLM connection failed")

        from agent import ChatBotAgent
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        results = list(agent.agent_tool_calling(messages=[]))
        assert len(results) == 1
        assert 'Error occurred' in results[0].content

    @patch('agent.WorkspaceClient')
    def test_predict_calls_predict_stream(self, mock_ws_cls):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from agent import ChatBotAgent
        from mlflow.pyfunc.model import ChatAgentChunk, ChatAgentMessage
        agent = ChatBotAgent(model_name='test', max_tokens=1000)

        mock_delta = ChatAgentMessage(role='assistant', content='Test response', id='msg-1')
        mock_chunk = ChatAgentChunk(delta=mock_delta)

        with patch.object(agent, 'predict_stream', return_value=iter([mock_chunk])):
            result = agent.predict(messages=[])
            assert len(result.messages) == 1


# =============================================================================
# SimpleChatBotAgent tests
# =============================================================================
class TestSimpleChatBotAgent:
    @patch('simple_agent.WorkspaceClient')
    def test_init(self, mock_ws_cls, sample_genie_spaces):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from simple_agent import SimpleChatBotAgent
        agent = SimpleChatBotAgent(
            model_name='databricks-claude-sonnet-4-6',
            max_tokens=1000,
            genie_spaces=sample_genie_spaces
        )
        assert len(agent.tools) == 3

    @patch('simple_agent.call_chat_model')
    @patch('simple_agent.WorkspaceClient')
    def test_simple_agent_stop(self, mock_ws_cls, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        response = MagicMock()
        response.id = 'resp-1'
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = 'stop'
        response.choices[0].message.role = 'assistant'
        response.choices[0].message.content = 'Simple answer'
        response.choices[0].message.tool_calls = None
        response.choices[0].message.to_dict.return_value = {
            'role': 'assistant', 'content': 'Simple answer'
        }
        mock_call.return_value = response

        from simple_agent import SimpleChatBotAgent
        agent = SimpleChatBotAgent(model_name='test', max_tokens=1000)

        results = list(agent.agent_tool_calling(messages=[]))
        assert len(results) == 1
        assert results[0].content == 'Simple answer'

    @patch('simple_agent.call_chat_model')
    @patch('simple_agent.run_genie')
    @patch('simple_agent.WorkspaceClient')
    def test_simple_agent_with_tool_call(self, mock_ws_cls, mock_run_genie, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        # First call: tool call response
        tc = MagicMock()
        tc.id = 'tc-1'
        tc.type = 'function'
        tc.function.name = 'genie_space123'
        tc.function.arguments = '{"prompt": "revenue?"}'

        tool_response = MagicMock()
        tool_response.id = 'resp-1'
        tool_response.choices = [MagicMock()]
        tool_response.choices[0].finish_reason = 'tool_calls'
        tool_response.choices[0].message.role = 'assistant'
        tool_response.choices[0].message.content = None
        tool_response.choices[0].message.tool_calls = [tc]

        # Second call: final answer
        final_response = MagicMock()
        final_response.id = 'resp-2'
        final_response.choices = [MagicMock()]
        final_response.choices[0].finish_reason = 'stop'
        final_response.choices[0].message.role = 'assistant'
        final_response.choices[0].message.content = 'Revenue is $1M'
        final_response.choices[0].message.tool_calls = None
        final_response.choices[0].message.to_dict.return_value = {
            'role': 'assistant', 'content': 'Revenue is $1M'
        }

        mock_call.side_effect = [tool_response, final_response]
        mock_run_genie.return_value = ('Revenue is $1M', 'conv-1')

        from simple_agent import SimpleChatBotAgent
        agent = SimpleChatBotAgent(model_name='test', max_tokens=1000, obo_token='tok')

        results = list(agent.agent_tool_calling(messages=[]))
        # Should yield: assistant tool_call message, tool result, final answer
        assert len(results) == 3
        assert results[-1].content == 'Revenue is $1M'

    @patch('simple_agent.call_chat_model')
    @patch('simple_agent.WorkspaceClient')
    def test_simple_agent_error_handling(self, mock_ws_cls, mock_call):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        mock_call.side_effect = Exception("Connection error")

        from simple_agent import SimpleChatBotAgent
        agent = SimpleChatBotAgent(model_name='test', max_tokens=1000)

        results = list(agent.agent_tool_calling(messages=[]))
        assert len(results) == 1
        assert 'Error occurred' in results[0].content

    @patch('simple_agent.WorkspaceClient')
    def test_predict_stream(self, mock_ws_cls):
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_ws_cls.return_value = mock_client

        from simple_agent import SimpleChatBotAgent
        from mlflow.pyfunc.model import ChatAgentMessage
        agent = SimpleChatBotAgent(model_name='test', max_tokens=1000)

        mock_msg = ChatAgentMessage(role='assistant', content='Streamed response', id='msg-1')

        with patch.object(agent, 'agent_tool_calling', return_value=iter([mock_msg])):
            chunks = list(agent.predict_stream(messages=[]))
            assert len(chunks) == 1
