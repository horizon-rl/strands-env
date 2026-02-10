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

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import tool

if TYPE_CHECKING:
    import boto3

CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Provides `execute_code` and `execute_command` tools for running Python code
    and shell commands in a sandboxed environment.

    Example:
        from strands_env.utils.aws import get_session

        session = get_session(region="us-east-1")
        toolkit = CodeInterpreterToolkit(boto3_session=session)

        # In environment:
        class MyEnv(Environment):
            def __init__(self, boto3_session, ...):
                self.toolkit = CodeInterpreterToolkit(boto3_session)

            def get_tools(self):
                return [self.toolkit.execute_code, self.toolkit.execute_command]

            async def cleanup(self):
                self.toolkit.cleanup()
    """

    def __init__(
        self,
        boto3_session: boto3.Session,
        session_name: str = "strands-env-session",
    ):
        """Initialize the toolkit.

        Args:
            boto3_session: boto3 session for AWS credentials.
            session_name: Name for the code interpreter session.
        """
        self.region = boto3_session.region_name
        self.session_name = session_name
        self._client = boto3_session.client("bedrock-agentcore", region_name=self.region)
        self._session_id: str | None = None
        self._execute_code = self._create_execute_code_tool()
        self._execute_command = self._create_execute_command_tool()

    def _get_session_id(self) -> str:
        """Get or create a code interpreter session."""
        if self._session_id is None:
            response = self._client.start_code_interpreter_session(
                codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                name=self.session_name,
                sessionTimeoutSeconds=3600,
            )
            self._session_id = response["sessionId"]
        return self._session_id

    def _parse_stream_response(self, response: dict[str, Any]) -> str:
        """Parse the EventStream response from invoke_code_interpreter.

        Extracts text content from result events or error messages from exceptions.
        Returns plain text that strands will wrap in tool result format.

        Args:
            response: Raw response from invoke_code_interpreter.

        Returns:
            Text content from execution result or error message.
        """
        errors: list[str] = []

        for event in response.get("stream", []):
            if "result" in event:
                result = event["result"]
                content = result.get("content", [])
                # Extract text from content list
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts) if texts else str(content)
                return str(content)

            # Check for exception events
            for error_key in (
                "accessDeniedException",
                "conflictException",
                "internalServerException",
                "resourceNotFoundException",
                "serviceQuotaExceededException",
                "throttlingException",
                "validationException",
            ):
                if error_key in event:
                    msg = event[error_key].get("message", error_key)
                    errors.append(f"{error_key}: {msg}")
                    break

        # No result found - return collected errors or generic message
        return "\n".join(errors) if errors else "No result received"

    def _create_execute_code_tool(self):
        """Create execute_code tool."""
        client = self._client

        @tool
        def execute_code(code: str) -> str:
            """Execute Python code and return the result.

            Args:
                code: The Python code to execute.

            Returns:
                Execution output text or error message.
            """
            response = client.invoke_code_interpreter(
                codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                sessionId=self._get_session_id(),
                name="executeCode",
                arguments={"code": code, "language": "python"},
            )
            return self._parse_stream_response(response)

        return execute_code

    def _create_execute_command_tool(self):
        """Create execute_command tool."""
        client = self._client

        @tool
        def execute_command(command: str) -> str:
            """Execute a shell command and return the result.

            Args:
                command: The shell command to execute.

            Returns:
                Execution output text or error message.
            """
            response = client.invoke_code_interpreter(
                codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                sessionId=self._get_session_id(),
                name="executeCommand",
                arguments={"command": command},
            )
            return self._parse_stream_response(response)

        return execute_command

    @property
    def execute_code(self):
        """Get the execute_code tool."""
        return self._execute_code

    @property
    def execute_command(self):
        """Get the execute_command tool."""
        return self._execute_command

    def cleanup(self) -> None:
        """Clean up code interpreter session."""
        if self._session_id:
            try:
                self._client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    sessionId=self._session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            self._session_id = None
