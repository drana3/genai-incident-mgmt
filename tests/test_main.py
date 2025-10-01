import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
import uuid
import json

client = TestClient(app)


# @pytest.mark.asyncio
# async def test_process_alert_success(mock_bedrock, mock_opensearch, mock_cross_encoder, mock_dynamodb, mock_ssm):
#     # Mock Crew output: high confidence, executor executed
#     mock_crew_output = json.dumps({
#         "issue": "Database timeout",
#         "root_cause": "High load",
#         "impact": "API delays",
#         "resolution": "Scale RDS",
#         "confidence": 0.85,
#         "executed": True,
#         "details": "SSM command executed"
#     })

#     with patch("app.tools.boto3.client", side_effect=[mock_bedrock, mock_ssm]), \
#             patch("app.main.boto3.client", side_effect=[mock_bedrock]), \
#             patch("app.tools.OpenSearch", return_value=mock_opensearch), \
#             patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder), \
#             patch("app.main.get_audit_table", return_value=mock_dynamodb), \
#             patch("crewai.crew.Crew.kickoff", return_value=mock_crew_output):

#         response = client.post("/process_alert", json={
#             "description": "DB timeout",
#             "severity": "high",
#             "metrics": {"cpu": 90}
#         })

#         assert response.status_code == 200
#         data = response.json()
#         assert data["status"] == "resolved"
#         assert "incident_id" in data

#         resolution = data["resolution"]
#         assert resolution["issue"] == "Database timeout"
#         assert "confidence" in resolution
#         assert float(resolution["confidence"]) >= 0.8
#         assert "executed" in resolution


@pytest.mark.asyncio
async def test_process_alert_low_confidence(mock_bedrock, mock_opensearch, mock_cross_encoder, mock_dynamodb, mock_ssm):
    # Mock Crew output: low confidence, executor skipped
    mock_crew_output = json.dumps({
        "issue": "Database timeout",
        "root_cause": "High load",
        "impact": "API delays",
        "resolution": "Scale RDS",
        "confidence": 0.7,
        "executed": False,
        "details": "Skipped execution, awaiting approval"
    })

    with patch("app.tools.boto3.client", side_effect=[mock_bedrock, mock_ssm]), \
            patch("app.main.boto3.client", side_effect=[mock_bedrock]), \
            patch("app.tools.OpenSearch", return_value=mock_opensearch), \
            patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder), \
            patch("app.main.get_audit_table", return_value=mock_dynamodb), \
            patch("crewai.crew.Crew.kickoff", return_value=mock_crew_output):

        response = client.post("/process_alert", json={
            "description": "DB timeout",
            "severity": "high"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending_human"

        resolution = data["resolution"]
        assert "confidence" in resolution
        assert float(resolution["confidence"]) < 0.8
        assert "executed" in resolution
        assert resolution["executed"] is False


@pytest.mark.asyncio
async def test_process_alert_empty_runbooks(mock_bedrock, mock_opensearch, mock_dynamodb, mock_ssm):
    # OpenSearch returns no hits
    mock_opensearch.search.return_value = {"hits": {"hits": []}}
    mock_crew_output = "{}"

    with patch("app.tools.boto3.client", side_effect=[mock_bedrock, mock_ssm]), \
            patch("app.main.boto3.client", side_effect=[mock_bedrock]), \
            patch("app.tools.OpenSearch", return_value=mock_opensearch), \
            patch("app.main.get_audit_table", return_value=mock_dynamodb), \
            patch("crewai.crew.Crew.kickoff", return_value=mock_crew_output):

        response = client.post("/process_alert", json={
            "description": "DB timeout",
            "severity": "high"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending_human"
        assert "confidence" in data["resolution"]
        assert "executed" in data["resolution"]


@pytest.mark.asyncio
async def test_approve_endpoint_accept(mock_dynamodb):
    with patch("app.main.get_audit_table", return_value=mock_dynamodb):
        incident_id = str(uuid.uuid4())
        response = client.post(
            f"/approve/{incident_id}",
            json={"approved": True, "tweaks": {"resolution": "Scale RDS"}}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "approved"


@pytest.mark.asyncio
async def test_approve_endpoint_reject(mock_dynamodb):
    with patch("app.main.get_audit_table", return_value=mock_dynamodb):
        incident_id = str(uuid.uuid4())
        response = client.post(
            f"/approve/{incident_id}",
            json={"approved": False}
        )
        assert response.status_code == 400
        assert "Approval rejected" in response.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_alert(mock_bedrock, mock_dynamodb, mock_ssm):
    with patch("app.tools.boto3.client", side_effect=[mock_bedrock, mock_ssm]), \
            patch("app.main.boto3.client", side_effect=[mock_bedrock]), \
            patch("app.main.get_audit_table", return_value=mock_dynamodb), \
            patch("crewai.crew.Crew.kickoff", return_value="{}"):

        response = client.post(
            "/process_alert", json={"description": "", "severity": "invalid"}
        )
        assert response.status_code == 422
