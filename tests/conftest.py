import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock
import boto3
from moto import mock_aws
import io
import json


# ----------------------------
# FastAPI test client
# ----------------------------
@pytest.fixture
def client():
    return TestClient(app)


# ----------------------------
# Bedrock mock (for embeddings + chat-like outputs)
# ----------------------------
@pytest.fixture
def mock_bedrock():
    bedrock = MagicMock()
    # For RAGTool: embedding response
    bedrock.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps({
            "embedding": [0.1, 0.2, 0.3]
        }).encode("utf-8"))
    }
    return bedrock


# ----------------------------
# OpenSearch mock (for RAGTool search)
# ----------------------------
@pytest.fixture
def mock_opensearch():
    opensearch = MagicMock()
    opensearch.search.return_value = {
        "hits": {
            "hits": [
                {"_source": {"content": "runbook-db-timeout"}},
                {"_source": {"content": "runbook-other"}}
            ]
        }
    }
    return opensearch

# ----------------------------
# CrossEncoder mock (for reranking in RAGTool)
# ----------------------------


@pytest.fixture
def mock_cross_encoder():
    mock = MagicMock()
    mock.predict = MagicMock(return_value=[0.9, 0.1])
    return mock

# ----------------------------
# DynamoDB mock (Moto)
# ----------------------------


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="IncidentAudit",
            KeySchema=[
                {"AttributeName": "incident_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "incident_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield table


# ----------------------------
# SSM mock (for SSMExecuteTool)
# ----------------------------
@pytest.fixture
def mock_ssm():
    with mock_aws():
        ssm = boto3.client("ssm", region_name="us-east-1")
        ssm.send_command = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}
        yield ssm
