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


def _create_bedrock_client(role_arn: str, region: Optional[str] = None, read_timeout: int = 600):
    """Create a bedrock-runtime boto3 client with auto-refreshing assumed role credentials."""
    import os
    import boto3
    from botocore.config import Config
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session

    botocore_session = get_session()

    def _refresh():
        # Use base EC2 instance credentials (not the assumed role)
        saved_profile = os.environ.pop("AWS_PROFILE", None)
        try:
            sts = boto3.Session(region_name=region).client("sts")
        finally:
            if saved_profile is not None:
                os.environ["AWS_PROFILE"] = saved_profile

        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="TradingAgents",
        )
        creds = resp["Credentials"]
        return {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["SessionToken"],
            "expiry_time": creds["Expiration"].isoformat(),
        }

    refreshable_creds = RefreshableCredentials.create_from_metadata(
        metadata=_refresh(),
        refresh_using=_refresh,
        method="sts-assume-role",
    )
    botocore_session._credentials = refreshable_creds

    session = boto3.Session(botocore_session=botocore_session, region_name=region)
    boto_config = Config(
        read_timeout=read_timeout,
        connect_timeout=10,
        retries={"max_attempts": 3},
    )
    return session.client("bedrock-runtime", config=boto_config)


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

        # Assume IAM role with auto-refreshing credentials
        role_arn = self.kwargs.get("role_arn")
        if role_arn:
            llm_kwargs["client"] = _create_bedrock_client(role_arn, region)

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
