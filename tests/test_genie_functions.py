"""
Tests for apps/genie_functions.py

Covers: run_genie, post_genie, create_message_genie, get_genie_message,
        extract_column_values_string, get_genie_name_by_id, GenieRunInput
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))

from genie_functions import (
    run_genie,
    post_genie,
    create_message_genie,
    get_genie_message,
    extract_column_values_string,
    get_genie_name_by_id,
    GenieRunInput,
)


# =============================================================================
# GenieRunInput validator tests
# =============================================================================
class TestGenieRunInput:
    def test_valid_input(self):
        inp = GenieRunInput(
            genie_space_id='space123',
            prompt='What are the sales?',
            obo_token='token-abc'
        )
        assert inp.genie_space_id == 'space123'
        assert inp.sleeper_time == 0.7
        assert inp.max_retries == 45

    def test_empty_space_id_rejected(self):
        with pytest.raises(Exception):
            GenieRunInput(genie_space_id='', prompt='test', obo_token='token')

    def test_empty_prompt_rejected(self):
        with pytest.raises(Exception):
            GenieRunInput(genie_space_id='space1', prompt='', obo_token='token')

    def test_empty_obo_token_rejected(self):
        with pytest.raises(Exception):
            GenieRunInput(genie_space_id='space1', prompt='test', obo_token='')

    def test_negative_sleeper_time_rejected(self):
        with pytest.raises(Exception):
            GenieRunInput(
                genie_space_id='space1', prompt='test', obo_token='token',
                sleeper_time=-1.0
            )

    def test_zero_max_retries_rejected(self):
        with pytest.raises(Exception):
            GenieRunInput(
                genie_space_id='space1', prompt='test', obo_token='token',
                max_retries=0
            )

    def test_custom_conversation_id(self):
        inp = GenieRunInput(
            genie_space_id='space1', prompt='test', obo_token='token',
            conversation_id='conv-123'
        )
        assert inp.conversation_id == 'conv-123'

    def test_none_conversation_id(self):
        inp = GenieRunInput(
            genie_space_id='space1', prompt='test', obo_token='token',
            conversation_id=None
        )
        assert inp.conversation_id is None


# =============================================================================
# get_genie_name_by_id tests
# =============================================================================
class TestGetGenieNameById:
    def test_found(self, sample_genie_spaces):
        result = get_genie_name_by_id(sample_genie_spaces, 'space123')
        assert result == 'Sales Analytics'

    def test_not_found(self, sample_genie_spaces):
        result = get_genie_name_by_id(sample_genie_spaces, 'nonexistent')
        assert result is None

    def test_empty_spaces(self):
        result = get_genie_name_by_id({}, 'space123')
        assert result is None


# =============================================================================
# extract_column_values_string tests
# =============================================================================
class TestExtractColumnValuesString:
    def test_basic_extraction(self):
        response = MagicMock()
        col1 = MagicMock()
        col1.name = 'product'
        col2 = MagicMock()
        col2.name = 'revenue'
        response.statement_response.manifest.schema.columns = [col1, col2]
        response.statement_response.result.data_array = [
            ['Widget A', '1000'],
            ['Widget B', '2000']
        ]

        result = extract_column_values_string(response)
        assert len(result) == 2
        assert 'product: Widget A' in result[0]
        assert 'revenue: 1000' in result[0]

    def test_truncates_at_50_rows(self):
        response = MagicMock()
        col = MagicMock()
        col.name = 'id'
        response.statement_response.manifest.schema.columns = [col]
        response.statement_response.result.data_array = [[str(i)] for i in range(100)]

        result = extract_column_values_string(response)
        assert len(result) == 50

    def test_single_column(self):
        response = MagicMock()
        col = MagicMock()
        col.name = 'name'
        response.statement_response.manifest.schema.columns = [col]
        response.statement_response.result.data_array = [['Alice'], ['Bob']]

        result = extract_column_values_string(response)
        assert result[0] == 'name: Alice'
        assert result[1] == 'name: Bob'


# =============================================================================
# post_genie tests
# =============================================================================
class TestPostGenie:
    def test_calls_start_conversation(self):
        mock_w = MagicMock()
        mock_response = MagicMock()
        mock_w.genie.start_conversation.return_value = mock_response

        result = post_genie('space123', 'What are the sales?', mock_w)
        mock_w.genie.start_conversation.assert_called_once_with(
            space_id='space123',
            content='What are the sales?'
        )
        assert result == mock_response


# =============================================================================
# create_message_genie tests
# =============================================================================
class TestCreateMessageGenie:
    def test_calls_create_message(self):
        mock_w = MagicMock()
        mock_response = MagicMock()
        mock_w.genie.create_message.return_value = mock_response

        result = create_message_genie('space123', 'Follow up query', 'conv-001', mock_w)
        mock_w.genie.create_message.assert_called_once_with(
            space_id='space123',
            content='Follow up query',
            conversation_id='conv-001'
        )
        assert result == mock_response


# =============================================================================
# get_genie_message tests
# =============================================================================
class TestGetGenieMessage:
    def test_completed_with_query_result(self):
        mock_w = MagicMock()

        # Create mock response
        mock_response = MagicMock()
        mock_response.status.value = 'COMPLETED'

        # Create attachment with query
        attachment = MagicMock()
        attachment.attachment_id = 'att-1'
        attachment.text = None
        attachment.query.query = 'SELECT * FROM sales'
        attachment.query.description = 'Sales query'
        mock_response.attachments = [attachment]

        mock_w.genie.get_message.return_value = mock_response

        # Mock query result
        query_result = MagicMock()
        col = MagicMock()
        col.name = 'total'
        query_result.statement_response.manifest.schema.columns = [col]
        query_result.statement_response.result.data_array = [['100']]
        mock_w.genie.get_message_query_result_by_attachment.return_value = query_result

        result = get_genie_message('space123', mock_w, 'conv-1', 'msg-1')
        assert 'Sales query' in result or 'total: 100' in result

    def test_completed_with_text_only(self):
        mock_w = MagicMock()

        mock_response = MagicMock()
        mock_response.status.value = 'COMPLETED'

        attachment = MagicMock()
        attachment.attachment_id = 'att-1'
        attachment.text.content = 'Here is a text response'
        attachment.query = None
        mock_response.attachments = [attachment]

        mock_w.genie.get_message.return_value = mock_response

        result = get_genie_message('space123', mock_w, 'conv-1', 'msg-1')
        assert result == 'Here is a text response'

    def test_failed_status(self):
        mock_w = MagicMock()
        mock_response = MagicMock()
        mock_response.status.value = 'FAILED'
        mock_response.error.error = 'Query syntax error'
        mock_w.genie.get_message.return_value = mock_response

        result = get_genie_message('space123', mock_w, 'conv-1', 'msg-1')
        assert result == 'Query syntax error'

    def test_max_retries_exceeded(self):
        mock_w = MagicMock()
        mock_response = MagicMock()
        mock_response.status.value = 'EXECUTING'
        mock_w.genie.get_message.return_value = mock_response

        result = get_genie_message(
            'space123', mock_w, 'conv-1', 'msg-1',
            sleeper_time=0.01, max_retries=2
        )
        assert 'did not complete' in result

    def test_no_valid_attachment(self):
        mock_w = MagicMock()
        mock_response = MagicMock()
        mock_response.status.value = 'COMPLETED'

        attachment = MagicMock()
        attachment.attachment_id = None
        mock_response.attachments = [attachment]

        mock_w.genie.get_message.return_value = mock_response

        result = get_genie_message('space123', mock_w, 'conv-1', 'msg-1')
        assert "didn't provide any valid attachments" in result


# =============================================================================
# run_genie integration tests
# =============================================================================
class TestRunGenie:
    @patch('genie_functions.WorkspaceClient')
    @patch('genie_functions.Config')
    def test_new_conversation(self, mock_config_cls, mock_ws_cls):
        mock_config = MagicMock()
        mock_config.host = 'https://test.databricks.com'
        mock_config_cls.return_value = mock_config

        mock_w = MagicMock()
        mock_ws_cls.return_value = mock_w

        # Mock start_conversation
        start_resp = MagicMock()
        start_resp.conversation_id = 'new-conv-1'
        start_resp.message_id = 'msg-1'
        mock_w.genie.start_conversation.return_value = start_resp

        # Mock get_message
        msg_resp = MagicMock()
        msg_resp.status.value = 'COMPLETED'
        attachment = MagicMock()
        attachment.attachment_id = 'att-1'
        attachment.text.content = 'Revenue is $1M'
        attachment.query = None
        msg_resp.attachments = [attachment]
        mock_w.genie.get_message.return_value = msg_resp

        result, conv_id = run_genie(
            genie_space_id='space123',
            prompt='What is revenue?',
            obo_token='test-token'
        )
        assert result == 'Revenue is $1M'
        assert conv_id == 'new-conv-1'

    @patch('genie_functions.WorkspaceClient')
    @patch('genie_functions.Config')
    def test_existing_conversation(self, mock_config_cls, mock_ws_cls):
        mock_config = MagicMock()
        mock_config.host = 'https://test.databricks.com'
        mock_config_cls.return_value = mock_config

        mock_w = MagicMock()
        mock_ws_cls.return_value = mock_w

        # Mock create_message (existing conversation)
        msg_resp = MagicMock()
        msg_resp.message_id = 'msg-2'
        mock_w.genie.create_message.return_value = msg_resp

        # Mock get_message
        get_resp = MagicMock()
        get_resp.status.value = 'COMPLETED'
        attachment = MagicMock()
        attachment.attachment_id = 'att-1'
        attachment.text.content = 'Updated result'
        attachment.query = None
        get_resp.attachments = [attachment]
        mock_w.genie.get_message.return_value = get_resp

        result, conv_id = run_genie(
            genie_space_id='space123',
            prompt='Follow up question',
            obo_token='test-token',
            conversation_id='existing-conv-1'
        )
        assert result == 'Updated result'
        assert conv_id == 'existing-conv-1'

    @patch('genie_functions.WorkspaceClient')
    @patch('genie_functions.Config')
    def test_error_handling(self, mock_config_cls, mock_ws_cls):
        mock_config = MagicMock()
        mock_config.host = 'https://test.databricks.com'
        mock_config_cls.return_value = mock_config

        mock_w = MagicMock()
        mock_ws_cls.return_value = mock_w
        mock_w.genie.start_conversation.side_effect = Exception("Network error")

        result, conv_id = run_genie(
            genie_space_id='space123',
            prompt='test',
            obo_token='test-token'
        )
        assert 'Error during operating Genie' in result
