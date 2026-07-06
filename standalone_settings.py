"""Standalone model settings copied from the parent project's config defaults.

This file deliberately avoids importing the parent project's ``config.py`` or
reading ``PIPELINE_*`` environment variables.  The values below are the same
model/region settings that were present in the project config when this
standalone agent was created.
"""

AWS_REGION = "ap-south-1"
LANGUAGE_MODEL_ID = "qwen.qwen3-vl-235b-a22b"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
MAX_TOKENS_JUDGE = 4096

BEDROCK_READ_TIMEOUT_SECONDS = 600
BEDROCK_MAX_POOL_CONNECTIONS = 10
