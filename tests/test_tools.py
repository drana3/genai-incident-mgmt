import pytest
from unittest.mock import patch
from app.tools import rag_tool, ssm_tool
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_rag_retrieve_success(mock_bedrock, mock_opensearch):
    with patch('app.tools.boto3.client', return_value=mock_bedrock), \
            patch('app.tools.OpenSearch', return_value=mock_opensearch), \
            patch('app.tools.CrossEncoder') as MockCE:  # ✅ FIXED target

        instance = MockCE.return_value
        instance.predict = MagicMock(return_value=[0.9, 0.1])

        results = rag_tool._run("DB timeout")

        assert len(results) == 2
        assert "runbook-db-timeout" in results[0]
        instance.predict.assert_called_once()


@pytest.mark.asyncio
async def test_rag_retrieve_empty(mock_bedrock, mock_opensearch):
    with patch('app.tools.boto3.client', side_effect=[mock_bedrock]), \
            patch('app.tools.OpenSearch', return_value=mock_opensearch):
        mock_opensearch.search.return_value = {'hits': {'hits': []}}
        results = rag_tool._run("Unknown issue")
        assert results == []


# @pytest.mark.asyncio
# async def test_ssm_execute_success(mock_ssm):
#     with patch('app.tools.boto3.client', return_value=mock_ssm):
#         command_id = ssm_tool._run("restart_service")
#         assert command_id == "cmd-123"
#         assert mock_ssm.send_command.called


# @pytest.mark.asyncio
# async def test_ssm_execute_failure(mock_ssm):
#     with patch('app.tools.boto3.client', return_value=mock_ssm):
#         mock_ssm.send_command.side_effect = Exception("SSM failed")
#         with pytest.raises(Exception, match="SSM failed"):
#             ssm_tool._run("restart_service")
