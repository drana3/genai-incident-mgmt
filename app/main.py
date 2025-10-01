import os
import time
import json
import uuid
import datetime
import logging
import re
from decimal import Decimal
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, field_validator
from enum import Enum
from dotenv import load_dotenv

import boto3
from botocore.exceptions import ClientError

# CrewAI + LiteLLM
from crewai import Agent, Task, Crew
from litellm.exceptions import RateLimitError, AuthenticationError, APIError

# Tools
from app.tools import rag_tool, ssm_tool

# ----------------------------
# Load env + logging
# ----------------------------
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("genai_incident")

os.environ.setdefault("LITELLM_PROVIDER", os.getenv(
    "LITELLM_PROVIDER", "bedrock"))

# ----------------------------
# Utils
# ----------------------------


def convert_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    return obj


def get_audit_table():
    return boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1")).Table(
        os.getenv("AUDIT_TABLE", "IncidentAudit")
    )


def assert_aws_credentials():
    try:
        sts = boto3.client("sts", region_name=os.getenv(
            "AWS_REGION", "us-east-1"))
        identity = sts.get_caller_identity()
        logger.info("✅ AWS identity OK: %s", identity.get("Arn"))
    except ClientError as e:
        logger.exception("AWS credentials invalid or expired")
        raise RuntimeError("Fix AWS creds or use AWS_PROFILE.") from e


try:
    assert_aws_credentials()
except RuntimeError as e:
    logger.warning("AWS auth check failed: %s", str(e))

# ----------------------------
# Models
# ----------------------------
MODEL_CLASSIFIER = os.getenv(
    "MODEL_CLASSIFIER", "anthropic.claude-3-haiku-20240307-v1:0")
MODEL_RAG = os.getenv("MODEL_RAG", "anthropic.claude-3-haiku-20240307-v1:0")
MODEL_ANALYZER = os.getenv(
    "MODEL_ANALYZER", "anthropic.claude-3-haiku-20240307-v1:0")
MODEL_EXECUTOR = os.getenv(
    "MODEL_EXECUTOR", "anthropic.claude-3-haiku-20240307-v1:0")

logger.info("🔧 Models: classifier=%s rag=%s analyzer=%s executor=%s",
            MODEL_CLASSIFIER, MODEL_RAG, MODEL_ANALYZER, MODEL_EXECUTOR)

# ----------------------------
# Retry wrapper
# ----------------------------


def exponential_backoff_retry(fn, retries=3, base_interval=1.0, max_interval=10.0,
                              allowed_exceptions=(RateLimitError, APIError)):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except allowed_exceptions as e:
            last_exc = e
            wait = min(max_interval, base_interval * (2 ** (attempt - 1)))
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.2fs",
                           attempt, retries, str(e), wait)
            time.sleep(wait)
        except AuthenticationError as e:
            logger.error("Auth error calling Bedrock: %s", e)
            raise
    raise last_exc

# ----------------------------
# Helper: Parse JSON safely
# ----------------------------


def _parse_possible_json(val: Any) -> Dict:
    if not val:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            pass
        try:
            match = re.search(r"\{.*\}", val, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            return {}
    return {}


# ----------------------------
# CrewAI Agents
# ----------------------------
classifier_agent = Agent(role="Incident Classifier", goal="Classify incident type",
                         backstory="Expert in identifying IT incident types", llm=MODEL_CLASSIFIER)

rag_agent = Agent(role="Runbook Selector", goal="Retrieve the most relevant runbook",
                  backstory="Expert in retrieving relevant runbooks", llm=MODEL_RAG, tools=[rag_tool])

analyzer_agent = Agent(role="Resolution Generator", goal="Generate resolution details",
                       backstory="Expert in analyzing incidents and proposing solutions", llm=MODEL_ANALYZER)

executor_agent = Agent(role="Fix Executor", goal="Execute automated fixes",
                       backstory="Specialist in automating IT resolutions", llm=MODEL_EXECUTOR, tools=[ssm_tool])

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="GenAI Incident Manager", version="1.0")


class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Alert(BaseModel):
    incident_id: Optional[str] = None
    description: constr(min_length=1)
    severity: SeverityEnum
    metrics: dict = {}

    @field_validator("description")
    def description_not_blank(cls, v):
        if not v.strip():
            raise ValueError("Description must not be blank")
        return v


class ApproveRequest(BaseModel):
    approved: bool
    tweaks: Optional[Dict] = {}

# ----------------------------
# Endpoint: process_alert
# ----------------------------


@app.post("/process_alert")
async def process_alert(alert: Alert):
    incident_id = alert.incident_id or str(uuid.uuid4())
    logger.info("🚨 Processing alert %s", incident_id)

    tasks = [
        Task(description=(
            "You are an **Incident Classifier Agent**. "
            "Follow these steps:\n"
            "1. Read the incident description carefully.\n"
            "2. Decide if it is most related to: (a) database, (b) network, or (c) application.\n"
            "3. If multiple categories seem possible, pick the most likely one (never say 'uncertain').\n"
            "4. Respond with exactly one lowercase word: `database`, `network`, or `application`.\n"
            "⚠️ Do not add explanations or extra text."
        ), expected_output="database OR network OR application", agent=classifier_agent),

        Task(description=(
            "You are a **Runbook Selector Agent**.\n"
            "Steps:\n"
            "1. Take the incident description.\n"
            "2. Call the RAG_Tool to retrieve the most relevant runbook(s).\n"
            "3. Always return the raw JSON runbook content exactly as retrieved.\n"
            "4. If no runbook is found, return `{}`.\n"
            "⚠️ Do not generate your own runbook; rely only on RAG_Tool results."
        ), expected_output="Runbook JSON content", agent=rag_agent),

        Task(description=(
            "You are a **Resolution Analyzer Agent**.\n"
            "Think step by step:\n"
            "1. Combine the incident type + runbook JSON.\n"
            "2. Identify the issue.\n"
            "3. Identify the most likely root cause.\n"
            "4. Assess the business/system impact.\n"
            "5. Suggest specific, actionable resolution steps.\n"
            "6. Assign a numeric confidence score (0–1).\n"
            "Return a **strict JSON** with keys:\n"
            "{\n"
            "  \"issue\": \"...\",\n"
            "  \"root_cause\": \"...\",\n"
            "  \"impact\": \"...\",\n"
            "  \"resolution\": \"...\",\n"
            "  \"confidence\": 0.xx\n"
            "}\n"
            "⚠️ No extra text outside the JSON."
        ), expected_output="Valid JSON {issue, root_cause, impact, resolution, confidence}", agent=analyzer_agent),

        Task(description=(
            "You are a **Fix Executor Agent**.\n"
            "Steps:\n"
            "1. Read the analyzer output JSON.\n"
            "2. If `confidence` > 0.8, call SSM_Execute tool to apply the fix.\n"
            "3. If confidence ≤ 0.8, skip execution.\n"
            "4. Always return valid JSON like:\n"
            "   - If executed: `{ \"executed\": true, \"command_id\": \"...\" }`\n"
            "   - If skipped: `{ \"executed\": false, \"note\": \"Skipped due to low confidence\" }`"
        ), expected_output="Execution result JSON", agent=executor_agent),
    ]

    crew = Crew(agents=[classifier_agent, rag_agent, analyzer_agent, executor_agent],
                tasks=tasks, verbose=True, process="sequential", max_concurrency=1)

    try:
        result = exponential_backoff_retry(
            lambda: crew.kickoff(inputs={"alert": alert.dict()}))
    except (RateLimitError, APIError) as e:
        logger.exception("❌ Bedrock rate limit/service error: %s", e)
        return {"status": "pending_human", "incident_id": incident_id,
                "resolution": {"issue": "LLM unavailable", "root_cause": "Rate limited or service error",
                               "impact": "Unknown", "resolution": "Manual investigation required",
                               "confidence": 0.0, "executed": False}}

    # ---------------- Extract outputs ----------------
    parsed_analyzer, parsed_executor = {}, {}

    try:
        for t in getattr(result, "tasks_output", []):
            agent_name = getattr(t, "agent", "") or ""
            raw_output = getattr(t, "raw", None) or getattr(t, "output", None)
            if not raw_output:
                raw_output = str(t)
            logger.info("📥 Agent %s raw output: %s", agent_name, raw_output)
            parsed_json = _parse_possible_json(raw_output)
            if "Resolution Generator" in agent_name:
                parsed_analyzer = parsed_json
            if "Fix Executor" in agent_name:
                parsed_executor = parsed_json
    except Exception as e:
        logger.warning("⚠️ Failed to extract task outputs: %s", e)

    # ---------------- Merge resolution ----------------
    resolution = {"issue": "Unknown", "root_cause": "Unknown", "impact": "Unknown",
                  "resolution": "Manual investigation required", "confidence": 0.7, "executed": False}

    if parsed_analyzer:
        resolution.update(parsed_analyzer)
    if parsed_executor:
        resolution.update(parsed_executor)

    if isinstance(resolution.get("command_id"), list):
        resolution["command_id"] = ",".join(resolution["command_id"])

    # ---------------- Final decision ----------------
    confidence = float(resolution.get("confidence", 0.7))
    if resolution.get("executed") is True:
        status = "resolved"
        resolution["confidence"] = max(confidence, 0.95)
    elif confidence < 0.8:
        status = "pending_human"
    else:
        status = "resolved"

    get_audit_table().put_item(Item=convert_floats({
        "incident_id": incident_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "resolution": resolution,
        "actions": str(result),
        "human_intervention": (status == "pending_human"),
    }))

    return {"status": status, "incident_id": incident_id, "resolution": resolution}

# ----------------------------
# Approve endpoint
# ----------------------------


@app.post("/approve/{incident_id}")
async def approve(incident_id: str, request: ApproveRequest):
    if not request.approved:
        raise HTTPException(status_code=400, detail="Approval rejected")
    return {"status": "approved", "incident_id": incident_id, "tweaks": request.tweaks}

# ----------------------------
# Lambda handler
# ----------------------------
try:
    from mangum import Mangum
    handler = Mangum(app)
except Exception:
    handler = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
