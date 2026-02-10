# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for AWS boto3 session."""

from __future__ import annotations

import logging
from functools import lru_cache

import boto3

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_session(
    region: str = "us-east-1",
    profile_name: str | None = None,
    role_arn: str | None = None,
    session_name: str = "StrandsEnvSession",
) -> boto3.Session:
    """Get a cached boto3 session.

    Credentials are managed by boto3's provider chain (env vars, ~/.aws/credentials,
    IAM instance role, etc.) and auto-refresh automatically.

    If `role_arn` is provided, assumes the role using STS with auto-refreshing
    credentials via botocore's `RefreshableCredentials`.

    Args:
        region: AWS region name.
        profile_name: Optional AWS profile name from ~/.aws/config.
        role_arn: Optional ARN of the IAM role to assume.
        session_name: Session name for assumed role (only used if role_arn provided).

    Returns:
        Cached boto3 Session instance.
    """
    if role_arn:
        return _create_assumed_role_session(role_arn, region, session_name)
    else:
        logger.info(f"Creating boto3 session: region={region}, profile={profile_name}")
        return boto3.Session(region_name=region, profile_name=profile_name)


def _create_assumed_role_session(role_arn: str, region: str, session_name: str) -> boto3.Session:
    """Create a boto3 session with assumed role credentials."""
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session as get_botocore_session

    logger.info(f"Creating boto3 session with assumed role: role={role_arn}, region={region}")

    def refresh() -> dict:
        logger.info(f"Refreshing STS credentials for assumed role: {role_arn}")
        sts = boto3.client("sts", region_name=region)
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)["Credentials"]
        return {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["SessionToken"],
            "expiry_time": creds["Expiration"].isoformat(),
        }

    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method="sts-assume-role",
    )

    botocore_session = get_botocore_session()
    botocore_session._credentials = session_credentials
    return boto3.Session(botocore_session=botocore_session, region_name=region)


def clear_session_cache() -> None:
    """Clear all cached boto3 sessions."""
    get_session.cache_clear()
