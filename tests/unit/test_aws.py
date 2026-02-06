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

"""Tests for AWS session utilities."""

from unittest.mock import MagicMock, patch

import boto3

from strands_env.utils.aws import clear_sessions, get_assumed_role_session, get_boto3_session


class TestGetBoto3Session:
    """Tests for get_boto3_session."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sessions()

    def test_returns_session(self):
        """Should return a boto3 Session."""
        session = get_boto3_session(region="us-west-2")
        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    def test_cached_by_region(self):
        """Same region should return cached session."""
        session1 = get_boto3_session(region="us-east-1")
        session2 = get_boto3_session(region="us-east-1")
        assert session1 is session2

    def test_different_regions_different_sessions(self):
        """Different regions should return different sessions."""
        session1 = get_boto3_session(region="us-east-1")
        session2 = get_boto3_session(region="us-west-2")
        assert session1 is not session2

    @patch("strands_env.utils.aws.boto3.Session")
    def test_cached_by_profile(self, mock_session_cls):
        """Same profile should return cached session."""
        mock_session_cls.return_value = MagicMock()
        session1 = get_boto3_session(region="us-east-1", profile_name="test-profile")
        session2 = get_boto3_session(region="us-east-1", profile_name="test-profile")
        assert session1 is session2
        # Session should only be created once due to caching
        assert mock_session_cls.call_count == 1

    @patch("strands_env.utils.aws.boto3.Session")
    def test_different_profiles_different_sessions(self, mock_session_cls):
        """Different profiles should return different sessions."""
        mock_session_cls.side_effect = [MagicMock(), MagicMock()]
        session1 = get_boto3_session(region="us-east-1", profile_name="profile-a")
        session2 = get_boto3_session(region="us-east-1", profile_name="profile-b")
        assert session1 is not session2


class TestGetAssumedRoleSession:
    """Tests for get_assumed_role_session."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sessions()

    @patch("strands_env.utils.aws.boto3.client")
    @patch("botocore.session.get_session")
    def test_assumes_role(self, mock_get_session, mock_boto3_client):
        """Should call STS assume_role."""
        from datetime import datetime, timezone

        # Mock STS response
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA_TEST",
                "SecretAccessKey": "secret_test",
                "SessionToken": "token_test",
                "Expiration": datetime.now(timezone.utc),
            }
        }
        mock_boto3_client.return_value = mock_sts

        # Mock botocore session
        mock_botocore_session = MagicMock()
        mock_get_session.return_value = mock_botocore_session

        role_arn = "arn:aws:iam::123456789:role/TestRole"
        session = get_assumed_role_session(role_arn=role_arn, region="us-east-1")

        # Verify STS was called
        mock_boto3_client.assert_called_with("sts", region_name="us-east-1")
        mock_sts.assume_role.assert_called_with(RoleArn=role_arn, RoleSessionName="StrandsEnvSession")

        # Verify session was created
        assert session is not None

    @patch("strands_env.utils.aws.boto3.client")
    @patch("botocore.session.get_session")
    def test_cached_by_role_arn(self, mock_get_session, mock_boto3_client):
        """Same role ARN should return cached session."""
        from datetime import datetime, timezone

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA_TEST",
                "SecretAccessKey": "secret_test",
                "SessionToken": "token_test",
                "Expiration": datetime.now(timezone.utc),
            }
        }
        mock_boto3_client.return_value = mock_sts
        mock_get_session.return_value = MagicMock()

        role_arn = "arn:aws:iam::123456789:role/TestRole"
        session1 = get_assumed_role_session(role_arn=role_arn)
        session2 = get_assumed_role_session(role_arn=role_arn)

        assert session1 is session2
        # assume_role should only be called once due to caching
        assert mock_sts.assume_role.call_count == 1


class TestClearSessions:
    """Tests for clear_sessions."""

    def test_clears_all_caches(self):
        """Should clear both session caches."""
        # Create some cached sessions
        session1 = get_boto3_session(region="us-east-1")

        # Clear
        clear_sessions()

        # New call should create new session
        session2 = get_boto3_session(region="us-east-1")
        assert session1 is not session2
