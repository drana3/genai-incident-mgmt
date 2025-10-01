import boto3
import json
client = boto3.client("bedrock", region_name="us-east-1")
resp = client.list_foundation_models()
for m in resp["modelSummaries"]:
    print(m["modelId"], "->", m["providerName"])
