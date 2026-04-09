from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content

_PASSTHROUGH_KWARGS = (
    "max_retries", "max_tokens", "callbacks",
)


def _get_normalized_class():
    """Lazy import to avoid requiring langchain-aws when not using Bedrock."""
    from langchain_aws import ChatBedrockConverse

    class NormalizedChatBedrockConverse(ChatBedrockConverse):
        """ChatBedrockConverse with normalized content output."""

        def invoke(self, input, config=None, **kwargs):
            return normalize_content(super().invoke(input, config, **kwargs))

    return NormalizedChatBedrockConverse


def _assume_role(role_arn: str, region: Optional[str] = None) -> dict:
    """Assume an IAM role via STS using base EC2 instance credentials."""
    import os
    import boto3
    # Temporarily clear AWS_PROFILE so boto3 falls back to the EC2 instance
    # metadata credentials instead of an already-assumed role.
    saved_profile = os.environ.pop("AWS_PROFILE", None)
    try:
        session = boto3.Session(region_name=region)
        sts = session.client("sts")
    finally:
        if saved_profile is not None:
            os.environ["AWS_PROFILE"] = saved_profile
    resp = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="TradingAgents",
    )
    creds = resp["Credentials"]
    return {
        "aws_access_key_id": creds["AccessKeyId"],
        "aws_secret_access_key": creds["SecretAccessKey"],
        "aws_session_token": creds["SessionToken"],
    }


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatBedrockConverse instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model_id": self.model}

        region = self.kwargs.get("region_name")
        if region:
            llm_kwargs["region_name"] = region

        # Assume IAM role if role_arn is provided
        role_arn = self.kwargs.get("role_arn")
        if role_arn:
            creds = _assume_role(role_arn, region)
            llm_kwargs.update(creds)

        # Generous timeout for large models like Opus 4.6
        llm_kwargs["timeout"] = self.kwargs.get("timeout", 300)

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        cls = _get_normalized_class()
        return cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Accept any Bedrock model ID."""
        return True
