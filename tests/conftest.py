"""
Shared pytest fixtures for the Multi-Genie Agent test suite.

Provides mocks for Databricks SDK, OpenAI client, Flask request context,
and common test data structures used across all test modules.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import queue
import threading

# Ensure the apps directory is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))

# Set required environment variables before any app imports
os.environ.setdefault('SERVING_ENDPOINT', 'databricks-claude-sonnet-4-6')
os.environ.setdefault('DATABRICKS_HOST', 'https://test.databricks.com')
os.environ.setdefault('DATABRICKS_TOKEN', 'test-token')


@pytest.fixture
def mock_workspace_client():
    """Mocked Databricks WorkspaceClient."""
    with patch('databricks.sdk.WorkspaceClient') as mock_cls:
        mock_client = MagicMock()
        mock_client.serving_endpoints.get_open_ai_client.return_value = MagicMock()
        mock_client.genie = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Mocked OpenAI-compatible client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_config():
    """Mocked Databricks SDK Config."""
    with patch('databricks.sdk.core.Config') as mock_cls:
        mock_cfg = MagicMock()
        mock_cfg.host = 'https://test.databricks.com'
        mock_cls.return_value = mock_cfg
        yield mock_cfg


@pytest.fixture
def sample_genie_spaces():
    """Sample genie spaces dictionary for testing."""
    return {
        'Sales Analytics': {
            'id': 'space123',
            'description': 'Analytics for sales data'
        },
        'HR Dashboard': {
            'id': 'space456',
            'description': 'Human resources metrics'
        },
        'No Description Space': {
            'id': 'space789',
            'description': ''
        }
    }


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi there! How can I help?'},
        {'role': 'user', 'content': 'What are the sales numbers?'},
        {'role': 'assistant', 'content': 'Let me check with the Sales Genie.'}
    ]


@pytest.fixture
def sample_messages():
    """Sample messages list for LLM calls."""
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the total revenue?'}
    ]


@pytest.fixture
def mock_llm_response():
    """Mocked LLM chat completion response (no tool calls)."""
    response = MagicMock()
    response.id = 'resp-123'
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = 'stop'
    response.choices[0].message.role = 'assistant'
    response.choices[0].message.content = 'The total revenue is $1.2M.'
    response.choices[0].message.tool_calls = None
    response.choices[0].message.to_dict.return_value = {
        'role': 'assistant',
        'content': 'The total revenue is $1.2M.'
    }
    return response


@pytest.fixture
def mock_llm_tool_response():
    """Mocked LLM response with tool calls."""
    response = MagicMock()
    response.id = 'resp-456'
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = 'tool_calls'
    response.choices[0].message.role = 'assistant'
    response.choices[0].message.content = None

    tool_call = MagicMock()
    tool_call.id = 'call-001'
    tool_call.type = 'function'
    tool_call.function.name = 'genie_space123'
    tool_call.function.arguments = '{"prompt": "What is the total revenue?"}'

    response.choices[0].message.tool_calls = [tool_call]
    return response


@pytest.fixture
def mock_flask_request():
    """Mocked Flask request context with OBO headers."""
    with patch('flask.request') as mock_req:
        mock_req.headers = {
            'X-Forwarded-Preferred-Username': 'test.user@company.com',
            'X-Forwarded-Access-Token': 'obo-test-token-123'
        }
        yield mock_req


@pytest.fixture
def sample_config_yml(tmp_path):
    """Creates a temporary config.yml file for testing."""
    config_content = """model_name: databricks-claude-sonnet-4-6
system_prompt: >
  You are a helpful AI assistant. Your main tools are Genie spaces.
  Before attempting to use any Genie Space, check if it exists in your available tools.

react_system_prompt: >
  You are a ReAct agent, helping the user answer questions.
  Your main tools are Genie spaces. Follow THOUGHT, ACTION, OBSERVATION, ANSWER format.
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def chunk_queue():
    """A pre-filled chunk queue for testing streaming."""
    q = queue.Queue()
    q.put("Hello ")
    q.put("world!")
    q.put("__STREAMING_COMPLETE__")
    return q
