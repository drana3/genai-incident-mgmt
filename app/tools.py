from botocore.exceptions import ClientError
import os
import json
import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from sentence_transformers import CrossEncoder
from crewai.tools import BaseTool

logger = logging.getLogger("RAG_Tool")
logger.setLevel(logging.INFO)


def get_opensearch_client():
    """
    Returns an OpenSearch client with SigV4 authentication.
    """
    host = os.getenv(
        "OPENSEARCH_URL")  # e.g. search-incident-mgmt-xxxx.us-east-1.es.amazonaws.com
    region = os.getenv("AWS_REGION", "us-east-1")

    if not host:
        raise RuntimeError("⚠️ OPENSEARCH_URL not set in .env")

    session = boto3.Session()
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, "es")

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


class RAGTool(BaseTool):
    name: str = "RAG_Tool"
    description: str = "Retrieve and rerank runbooks from OpenSearch"

    def _run(self, query: str) -> list:
        try:
            client = get_opensearch_client()

            # Step 1: Embed query with Bedrock Titan
            bedrock = boto3.client(
                "bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
            response = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": query}),
                contentType="application/json",
                accept="application/json"
            )
            resp_body = json.loads(response["body"].read().decode("utf-8"))
            query_embedding = resp_body.get("embedding", [])

            if not query_embedding:
                logger.warning("⚠️ No embedding returned for query")
                return []

            # Step 2: Search OpenSearch
            query_body = {
                "size": 10,
                "query": {
                    "knn": {
                        "vector_field": {
                            "vector": query_embedding,
                            "k": 10,
                        }
                    }
                }
            }

            search_response = client.search(index="runbooks", body=query_body)
            docs = [hit["_source"]["content"]
                    for hit in search_response["hits"]["hits"]]

            if not docs:
                logger.info("ℹ️ No runbooks found in OpenSearch for query")
                return []

            # Step 3: Rerank with CrossEncoder
            cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc) for doc in docs]
            scores = cross_encoder.predict(pairs)

            reranked = sorted(zip(docs, scores),
                              key=lambda x: x[1], reverse=True)[:3]

            return [doc for doc, _ in reranked]

        except Exception as e:
            logger.error("❌ RAG_Tool failed: %s", e, exc_info=True)
            return []


# ✅ Exported instance
rag_tool = RAGTool()


logger = logging.getLogger("SSM_Execute")
logger.setLevel(logging.INFO)


class SSMExecuteTool(BaseTool):
    name: str = "SSM_Execute"
    description: str = "Run SSM command on EC2 instances"

    def _run(self, command: str) -> str:
        # If running in TEST_MODE, simulate deterministic behavior
        if os.getenv("TEST_MODE", "false").lower() == "true":
            if command == "fail":
                raise Exception("Simulated failure in TEST_MODE")
            return "cmd-123"  # deterministic return for tests

        # Otherwise, actually call AWS SSM
        try:
            ssm = boto3.client("ssm", region_name="us-east-1")
            response = ssm.send_command(
                InstanceIds=["i-1234567890abcdef0"],  # Replace in production
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [command]},
            )
            return response["Command"]["CommandId"]
        except ClientError as e:
            raise Exception(f"SSM execution failed: {str(e)}") from e


ssm_tool = SSMExecuteTool()
