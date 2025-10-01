import boto3
import json
import os
import glob
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# AWS + OpenSearch Setup
# ----------------------------
region = os.getenv("AWS_REGION", "us-east-1")
service = "es"

# AWS Credentials (from ~/.aws/credentials, env vars, or IAM role)
session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token,
)

# OpenSearch domain hostname (NO https://)
# e.g., search-my-domain-xyz.us-east-1.es.amazonaws.com
opensearch_url = os.getenv("OPENSEARCH_URL")

client = OpenSearch(
    hosts=[{"host": opensearch_url, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

bedrock = boto3.client("bedrock-runtime", region_name=region)

INDEX_NAME = "runbooks"


# ----------------------------
# Helpers
# ----------------------------
def create_index_if_missing():
    """Create OpenSearch index with knn_vector mapping for Titan embeddings."""
    if not client.indices.exists(INDEX_NAME):
        print(f"📌 Creating index '{INDEX_NAME}' ...")
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {"index": {"knn": True}},  # enable kNN
                "mappings": {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": 1024  # Titan v2 output size
                        },
                        "id": {"type": "keyword"},
                        "issue": {"type": "text"},
                        "root_cause": {"type": "text"},
                        "impact": {"type": "text"},
                        "resolution_steps": {"type": "text"},
                        "content": {"type": "text"},
                    }
                },
            },
        )
        print("✅ Index created")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists")


def embed_text(text: str):
    """Generate Titan embedding for text."""
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
    )
    resp_body = json.loads(response["body"].read().decode("utf-8"))
    embedding = resp_body.get(
        "embedding") or resp_body.get("embeddings", [])[0]
    return embedding


def index_runbook(file_path: str):
    """Index a single runbook JSON into OpenSearch."""
    with open(file_path, "r") as f:
        runbook = json.load(f)

    runbook_id = runbook.get("id") or os.path.basename(file_path)
    text = json.dumps(runbook)

    embedding = embed_text(text)

    doc = {
        "id": runbook_id,
        "issue": runbook.get("issue", ""),
        "root_cause": runbook.get("root_cause", ""),
        "impact": runbook.get("impact", ""),
        "resolution_steps": "\n".join(runbook.get("resolution_steps", [])),
        "content": text,
        "vector_field": embedding,
    }

    client.index(index=INDEX_NAME, id=runbook_id, body=doc)
    print(f"✅ Indexed runbook {runbook_id}")


def index_all_runbooks():
    """Ingest all runbooks from app/runbooks/"""
    create_index_if_missing()
    files = glob.glob("app/runbooks/*.json")

    if not files:
        print("⚠️ No runbooks found in app/runbooks/")
        return

    for file in files:
        try:
            index_runbook(file)
        except Exception as e:
            print(f"❌ Failed to index {file}: {e}")


if __name__ == "__main__":
    index_all_runbooks()
