# import boto3
# import json

# # Create a Bedrock client
# client = boto3.client("bedrock-runtime", region_name="us-east-1")

# # Model ID from Bedrock
# model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# # Define input
# prompt = """
# You are an expert assistant.
# Explain the difference between OpenSearch and SentenceTransformers in simple words.
# """

# response = client.invoke_model(
#     modelId=model_id,
#     body=json.dumps({
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "max_tokens": 500,
#         "temperature": 0.7
#     })
# )

# # Parse response
# output = json.loads(response["body"].read())
# print(output["output"]["message"]["content"][0]["text"])
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

body = {
    "anthropic_version": "bedrock-2023-05-31",  # REQUIRED
    "max_tokens": 500,
    "messages": [
        {"role": "user", "content": "Explain vector search in OpenSearch in very simple terms."}
    ]
}

response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

result = json.loads(response["body"].read())

# Correct way to extract text
print(result["content"][0]["text"])
