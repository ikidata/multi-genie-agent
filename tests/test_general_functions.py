"""
Tests for apps/general_functions.py

Covers: clean_text, create_tools_from_genie_spaces, call_chat_model,
        prepare_messages_for_llm, create_tool_calls_output, run_rest_api,
        get_model_list, optimize_conversation_history, count_tokens,
        extract_assistant_and_tool_messages, ChatConfig, Pydantic validators
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))

from general_functions import (
    clean_text,
    create_tools_from_genie_spaces,
    call_chat_model,
    prepare_messages_for_llm,
    create_tool_calls_output,
    run_rest_api,
    get_model_list,
    optimize_conversation_history,
    count_tokens,
    extract_assistant_and_tool_messages,
    ChatConfig,
    ToolCall,
    ToolFunctionCall,
    ToolCallsOutput,
)


# =============================================================================
# clean_text tests
# =============================================================================
class TestCleanText:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_plain_text_unchanged(self):
        assert clean_text("Hello world") == "Hello world"

    def test_extracts_answer_content(self):
        text = "**THOUGHT** Thinking about it\n**ANSWER** The answer is 42."
        result = clean_text(text)
        assert "The answer is 42." in result
        assert "THOUGHT" not in result

    def test_removes_all_markers(self):
        text = "**THOUGHT** some thought\n**ACTION** do something\n**OBSERVATION** saw result\n**ANSWER** final answer"
        result = clean_text(text)
        assert "THOUGHT" not in result
        assert "ACTION" not in result
        assert "OBSERVATION" not in result
        assert "final answer" in result

    def test_answer_with_colon(self):
        text = "**ANSWER**: Here is the response"
        result = clean_text(text)
        assert "Here is the response" in result

    def test_answer_without_colon(self):
        text = "**ANSWER** Direct response"
        result = clean_text(text)
        assert "Direct response" in result

    def test_no_answer_marker_removes_other_markers(self):
        text = "**THOUGHT** Just a thought about this topic"
        result = clean_text(text)
        assert "Just a thought about this topic" in result
        assert "THOUGHT" not in result

    def test_invalid_input_type_returns_input(self):
        # clean_text has a bare except that catches TypeError and returns the input
        result = clean_text(123)
        assert result == 123

    def test_none_input_returns_none(self):
        # clean_text has a bare except that catches TypeError and returns the input
        result = clean_text(None)
        assert result is None

    def test_multiline_answer(self):
        text = "**ANSWER**\nLine 1\nLine 2\nLine 3"
        result = clean_text(text)
        assert "Line 1" in result
        assert "Line 3" in result

    def test_nested_markers(self):
        text = "**THOUGHT** before **ANSWER** the real answer"
        result = clean_text(text)
        assert "the real answer" in result

    def test_incomplete_markers(self):
        text = "**THOUGHT some text without closing"
        result = clean_text(text)
        assert "some text without closing" in result


# =============================================================================
# create_tools_from_genie_spaces tests
# =============================================================================
class TestCreateToolsFromGenieSpaces:
    def test_basic_tool_creation(self, sample_genie_spaces):
        tools = create_tools_from_genie_spaces(sample_genie_spaces)
        assert len(tools) == 3
        assert all(t['type'] == 'function' for t in tools)

    def test_tool_names_use_genie_ids(self, sample_genie_spaces):
        tools = create_tools_from_genie_spaces(sample_genie_spaces)
        tool_names = [t['function']['name'] for t in tools]
        assert 'genie_space123' in tool_names
        assert 'genie_space456' in tool_names

    def test_tool_description_fallback(self, sample_genie_spaces):
        tools = create_tools_from_genie_spaces(sample_genie_spaces)
        # Find the tool with no description
        no_desc_tool = [t for t in tools if t['function']['name'] == 'genie_space789'][0]
        assert 'Genie space for No Description Space' in no_desc_tool['function']['description']

    def test_tool_has_prompt_parameter(self, sample_genie_spaces):
        tools = create_tools_from_genie_spaces(sample_genie_spaces)
        for tool in tools:
            params = tool['function']['parameters']
            assert 'prompt' in params['properties']
            assert 'prompt' in params['required']

    def test_empty_genie_spaces(self):
        tools = create_tools_from_genie_spaces({})
        assert tools == []

    def test_single_genie_space(self):
        spaces = {'TestSpace': {'id': 'abc', 'description': 'Test description'}}
        tools = create_tools_from_genie_spaces(spaces)
        assert len(tools) == 1
        assert tools[0]['function']['name'] == 'genie_abc'
        assert tools[0]['function']['description'] == 'Test description'


# =============================================================================
# call_chat_model tests
# =============================================================================
class TestCallChatModel:
    def test_successful_call(self, mock_openai_client, mock_llm_response):
        mock_openai_client.chat.completions.create.return_value = mock_llm_response

        # Create mock messages with model_dump_compat
        msg = MagicMock()
        msg.model_dump_compat.return_value = {'role': 'user', 'content': 'Hello'}
        messages = [msg]

        result = call_chat_model(mock_openai_client, 'test-model', messages, max_tokens=500)
        assert result == mock_llm_response
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_removes_empty_tool_list(self, mock_openai_client, mock_llm_response):
        mock_openai_client.chat.completions.create.return_value = mock_llm_response

        msg = MagicMock()
        msg.model_dump_compat.return_value = {'role': 'user', 'content': 'Hello'}

        call_chat_model(mock_openai_client, 'test-model', [msg], tool=[])
        call_args = mock_openai_client.chat.completions.create.call_args
        assert 'tool' not in call_args.kwargs

    def test_api_error_raises_runtime_error(self, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")

        msg = MagicMock()
        msg.model_dump_compat.return_value = {'role': 'user', 'content': 'Hello'}

        with pytest.raises(RuntimeError, match="Model endpoint calling error"):
            call_chat_model(mock_openai_client, 'test-model', [msg])

    def test_passes_tools_kwarg(self, mock_openai_client, mock_llm_response):
        mock_openai_client.chat.completions.create.return_value = mock_llm_response
        tools = [{'type': 'function', 'function': {'name': 'test'}}]

        msg = MagicMock()
        msg.model_dump_compat.return_value = {'role': 'user', 'content': 'Hello'}

        call_chat_model(mock_openai_client, 'test-model', [msg], tools=tools)
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs.get('tools') == tools


# =============================================================================
# prepare_messages_for_llm tests
# =============================================================================
class TestPrepareMessagesForLlm:
    def test_filters_compatible_keys(self):
        msg = MagicMock()
        msg.model_dump_compat.return_value = {
            'role': 'user',
            'content': 'Hello',
            'name': None,
            'id': 'msg-123',
            'attachments': None,
            'tool_calls': None,
            'tool_call_id': None
        }
        result = prepare_messages_for_llm([msg])
        assert len(result) == 1
        assert 'role' in result[0]
        assert 'content' in result[0]
        assert 'id' not in result[0]  # id is not in compatible_keys
        assert 'attachments' not in result[0]  # filtered out and None

    def test_excludes_none_values(self):
        # model_dump_compat is called with exclude_none=True, so the mock should reflect that
        msg = MagicMock()
        msg.model_dump_compat.return_value = {
            'role': 'user',
            'content': 'Hello'
        }
        result = prepare_messages_for_llm([msg])
        assert 'name' not in result[0]
        assert 'tool_calls' not in result[0]
        assert result[0] == {'role': 'user', 'content': 'Hello'}

    def test_model_dump_fallback(self):
        """Test fallback to model_dump when model_dump_compat is not available."""
        msg = MagicMock(spec=[])  # Empty spec so model_dump_compat doesn't exist
        msg.model_dump = MagicMock(return_value={
            'role': 'user',
            'content': 'Hello',
            'name': None,
            'id': 'msg-1',
            'tool_calls': None,
            'tool_call_id': None,
            'attachments': None
        })
        result = prepare_messages_for_llm([msg])
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert 'name' not in result[0]
        assert 'id' not in result[0]

    def test_dict_message_input(self):
        """Test handling of plain dict messages."""
        msg = {'role': 'user', 'content': 'Hello', 'id': '123', 'extra': 'field'}
        result = prepare_messages_for_llm([msg])
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert 'id' not in result[0]
        assert 'extra' not in result[0]


# =============================================================================
# create_tool_calls_output tests
# =============================================================================
class TestCreateToolCallsOutput:
    def test_basic_tool_call_output(self):
        tool_call = MagicMock()
        tool_call.id = 'call-001'
        tool_call.type = 'function'
        tool_call.function.name = 'genie_space123'
        tool_call.function.arguments = '{"prompt": "test"}'

        results = MagicMock()
        results.tool_calls = [tool_call]

        output = create_tool_calls_output(results)
        assert len(output) == 1
        assert output[0]['id'] == 'call-001'
        assert output[0]['function']['name'] == 'genie_space123'

    def test_dict_arguments_converted_to_json(self):
        tool_call = MagicMock()
        tool_call.id = 'call-002'
        tool_call.type = 'function'
        tool_call.function.name = 'genie_space456'
        tool_call.function.arguments = {'prompt': 'test query'}

        results = MagicMock()
        results.tool_calls = [tool_call]

        output = create_tool_calls_output(results)
        assert output[0]['function']['arguments'] == '{"prompt": "test query"}'

    def test_multiple_tool_calls(self):
        tc1 = MagicMock()
        tc1.id = 'call-001'
        tc1.type = 'function'
        tc1.function.name = 'genie_a'
        tc1.function.arguments = '{"prompt": "q1"}'

        tc2 = MagicMock()
        tc2.id = 'call-002'
        tc2.type = 'function'
        tc2.function.name = 'genie_b'
        tc2.function.arguments = '{"prompt": "q2"}'

        results = MagicMock()
        results.tool_calls = [tc1, tc2]

        output = create_tool_calls_output(results)
        assert len(output) == 2


# =============================================================================
# run_rest_api tests
# =============================================================================
class TestRunRestApi:
    @patch('general_functions.requests.Session')
    def test_successful_get(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'result': 'success'}
        mock_session_cls.return_value.request.return_value = mock_resp

        result = run_rest_api(
            server_hostname='https://test.databricks.com',
            token='test-token',
            api_version='2.0',
            api_command='genie/spaces',
            action_type='GET'
        )
        assert result == mock_resp

    @patch('general_functions.requests.Session')
    def test_successful_post(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'id': '123'}
        mock_session_cls.return_value.request.return_value = mock_resp

        result = run_rest_api(
            server_hostname='https://test.databricks.com',
            token='test-token',
            api_version='2.0',
            api_command='apps',
            action_type='POST',
            payload={'name': 'test-app'}
        )
        assert result == mock_resp

    def test_invalid_action_type(self):
        result = run_rest_api(
            server_hostname='https://test.databricks.com',
            token='test-token',
            api_version='2.0',
            api_command='test',
            action_type='DELETE'
        )
        assert isinstance(result, Exception)


# =============================================================================
# ChatConfig validator tests
# =============================================================================
class TestChatConfig:
    def test_valid_config(self):
        config = ChatConfig(model_name='databricks-claude-sonnet-4-6')
        assert config.model_name == 'databricks-claude-sonnet-4-6'

    def test_empty_model_name_rejected(self):
        with pytest.raises(Exception):
            ChatConfig(model_name='')

    def test_whitespace_model_name_rejected(self):
        with pytest.raises(Exception):
            ChatConfig(model_name='   ')

    def test_short_model_name_rejected(self):
        with pytest.raises(Exception):
            ChatConfig(model_name='a')


# =============================================================================
# get_model_list tests
# =============================================================================
class TestGetModelList:
    def test_default_model_returns_list(self, tmp_path):
        config_content = "model_name: databricks-claude-sonnet-4-6\n"
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        with patch('general_functions.open', mock_open(read_data=config_content)):
            with patch('yaml.safe_load', return_value={'model_name': 'databricks-claude-sonnet-4-6'}):
                result = get_model_list()
                assert isinstance(result, list)
                assert len(result) >= 1
                assert any(item['value'] == 'databricks-claude-sonnet-4-6' for item in result)

    def test_custom_model_added_first(self, tmp_path):
        config_content = "model_name: custom-model-v1\n"

        with patch('general_functions.open', mock_open(read_data=config_content)):
            with patch('yaml.safe_load', return_value={'model_name': 'custom-model-v1'}):
                result = get_model_list()
                assert result[0]['value'] == 'custom-model-v1'

    def test_missing_config_file(self):
        with patch('general_functions.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                get_model_list()


# =============================================================================
# optimize_conversation_history tests
# =============================================================================
class TestOptimizeConversationHistory:
    def test_empty_messages(self):
        assert optimize_conversation_history([]) == []

    @patch('general_functions.count_tokens', return_value=50)
    def test_preserves_system_message(self, mock_ct, sample_chat_history):
        result = optimize_conversation_history(sample_chat_history)
        assert result[0]['role'] == 'system'

    @patch('general_functions.count_tokens', return_value=50)
    def test_respects_default_message_count(self, mock_ct):
        messages = [{'role': 'system', 'content': 'sys'}]
        for i in range(30):
            messages.append({'role': 'user', 'content': f'msg {i}'})
        result = optimize_conversation_history(messages, default_messages=10)
        # System message + 10 most recent
        assert len(result) == 11

    @patch('general_functions.count_tokens', return_value=100000)
    def test_keeps_minimum_messages(self, mock_ct):
        messages = [{'role': 'system', 'content': 'sys'}]
        for i in range(10):
            messages.append({'role': 'user', 'content': f'msg {i}' * 1000})
        result = optimize_conversation_history(messages, max_tokens=100, min_messages=3)
        # At minimum: system + 3 messages
        assert len(result) >= 4

    @patch('general_functions.count_tokens', return_value=50)
    def test_no_system_message(self, mock_ct):
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi'}
        ]
        result = optimize_conversation_history(messages)
        assert len(result) == 2

    @patch('general_functions.count_tokens', return_value=50)
    def test_small_conversation_unchanged(self, mock_ct, sample_chat_history):
        result = optimize_conversation_history(sample_chat_history)
        assert len(result) == len(sample_chat_history)


# =============================================================================
# count_tokens tests
# =============================================================================
class TestCountTokens:
    @patch('general_functions.tiktoken.get_encoding')
    def test_empty_messages(self, mock_get_enc):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = []
        mock_get_enc.return_value = mock_enc
        assert count_tokens([]) == 0

    @patch('general_functions.tiktoken.get_encoding')
    def test_single_message(self, mock_get_enc):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens
        mock_get_enc.return_value = mock_enc
        messages = [{'role': 'user', 'content': 'Hello world'}]
        result = count_tokens(messages)
        assert result > 0

    @patch('general_functions.tiktoken.get_encoding')
    def test_longer_message_more_tokens(self, mock_get_enc):
        mock_enc = MagicMock()
        # Return different lengths based on input
        mock_enc.encode.side_effect = lambda s: list(range(len(s.split())))
        mock_get_enc.return_value = mock_enc
        short = [{'role': 'user', 'content': 'Hello'}]
        long = [{'role': 'user', 'content': 'Hello world this is longer'}]
        assert count_tokens(long) > count_tokens(short)

    @patch('general_functions.tiktoken.get_encoding')
    def test_message_without_content(self, mock_get_enc):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = []
        mock_get_enc.return_value = mock_enc
        messages = [{'role': 'user'}]
        result = count_tokens(messages)
        assert result == 0

    @patch('general_functions.tiktoken.get_encoding')
    def test_multiple_messages(self, mock_get_enc, sample_chat_history):
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s: list(range(len(s.split())))
        mock_get_enc.return_value = mock_enc
        result = count_tokens(sample_chat_history)
        assert result > 0


# =============================================================================
# extract_assistant_and_tool_messages tests
# =============================================================================
class TestExtractAssistantAndToolMessages:
    def test_filters_correctly(self):
        msg1 = MagicMock()
        msg1.role = 'system'
        msg2 = MagicMock()
        msg2.role = 'user'
        msg3 = MagicMock()
        msg3.role = 'assistant'
        msg4 = MagicMock()
        msg4.role = 'tool'

        completion = MagicMock()
        completion.messages = [msg1, msg2, msg3, msg4]

        result = extract_assistant_and_tool_messages(completion)
        assert len(result) == 2
        assert result[0].role == 'assistant'
        assert result[1].role == 'tool'

    def test_no_assistant_messages(self):
        msg1 = MagicMock()
        msg1.role = 'user'
        msg2 = MagicMock()
        msg2.role = 'system'

        completion = MagicMock()
        completion.messages = [msg1, msg2]

        result = extract_assistant_and_tool_messages(completion)
        assert len(result) == 0


# =============================================================================
# Pydantic model tests
# =============================================================================
class TestPydanticModels:
    def test_tool_function_call(self):
        tfc = ToolFunctionCall(name='test_func', arguments='{"key": "value"}')
        assert tfc.name == 'test_func'
        parsed = tfc.get_parsed_arguments()
        assert parsed == {"key": "value"}

    def test_tool_call(self):
        tc = ToolCall(
            id='call-1',
            type='function',
            function=ToolFunctionCall(name='genie_abc', arguments='{"prompt": "test"}')
        )
        assert tc.id == 'call-1'
        assert tc.function.name == 'genie_abc'

    def test_tool_calls_output(self):
        tc = ToolCall(
            id='call-1',
            type='function',
            function=ToolFunctionCall(name='genie_abc', arguments='{"prompt": "test"}')
        )
        output = ToolCallsOutput(tool_calls=[tc])
        dumped = output.model_dump()
        assert len(dumped['tool_calls']) == 1
