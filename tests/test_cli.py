"""Tests for CLI commands."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from token_bowl_chat_agent.cli import _generate_llm_response, app

runner = CliRunner()


class TestSendCommand:
    """Tests for the send command."""

    def test_send_requires_api_key(self):
        """Test send command fails without API key."""
        result = runner.invoke(app, ["send", "Hello world"])

        assert result.exit_code == 1
        assert "Token Bowl Chat API key required" in result.stdout

    @patch.dict(os.environ, {}, clear=True)
    def test_send_requires_openrouter_key_when_model_provided(self):
        """Test send command fails without OpenRouter key when using --model."""
        result = runner.invoke(
            app,
            ["send", "Hello world", "--api-key", "test_key", "--model", "openai/gpt-4o"],
        )

        assert result.exit_code == 1
        assert "OpenRouter API key required when using --model" in result.stdout

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_direct_mode(self, mock_client_class):
        """Test send command in direct mode (no LLM)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.id = "msg_123"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["send", "Hello world", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        assert "Sent to room:" in result.stdout
        mock_client.send_message.assert_called_once_with("Hello world", to_username=None)

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_direct_mode_with_recipient(self, mock_client_class):
        """Test send command in direct mode with recipient."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello friend"
        mock_response.id = "msg_124"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["send", "Hello friend", "--api-key", "test_key", "--to", "alice"],
        )

        assert result.exit_code == 0
        assert "Sent DM to @alice:" in result.stdout
        mock_client.send_message.assert_called_once_with("Hello friend", to_username="alice")

    @patch("token_bowl_chat_agent.cli._generate_llm_response")
    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_llm_mode(self, mock_client_class, mock_generate):
        """Test send command in LLM mode."""
        # Setup mocks
        mock_generate.return_value = "Generated response from LLM"

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response from LLM"
        mock_response.id = "msg_125"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "send",
                "What is fantasy football?",
                "--api-key",
                "test_key",
                "--model",
                "openai/gpt-4o",
                "--openrouter-key",
                "test_openrouter_key",
            ],
        )

        assert result.exit_code == 0
        assert "Sent to room:" in result.stdout
        # Verify LLM was called with the prompt
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["prompt"] == "What is fantasy football?"
        assert call_kwargs["model_name"] == "openai/gpt-4o"
        assert call_kwargs["openrouter_api_key"] == "test_openrouter_key"
        # Verify the LLM response was sent
        mock_client.send_message.assert_called_once_with(
            "Generated response from LLM", to_username=None
        )

    @patch("token_bowl_chat_agent.cli._generate_llm_response")
    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_llm_mode_with_system_prompt(self, mock_client_class, mock_generate):
        """Test send command in LLM mode with custom system prompt."""
        # Setup mocks
        mock_generate.return_value = "Expert response"

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Expert response"
        mock_response.id = "msg_126"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "send",
                "Analyze my team",
                "--api-key",
                "test_key",
                "--model",
                "openai/gpt-4o",
                "--openrouter-key",
                "test_openrouter_key",
                "--system",
                "You are a fantasy football expert",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a fantasy football expert"

    @patch("token_bowl_chat_agent.cli._generate_llm_response")
    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_llm_mode_mcp_disabled(self, mock_client_class, mock_generate):
        """Test send command in LLM mode with MCP disabled."""
        # Setup mocks
        mock_generate.return_value = "Response without MCP"

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Response without MCP"
        mock_response.id = "msg_127"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "send",
                "Hello",
                "--api-key",
                "test_key",
                "--model",
                "openai/gpt-4o",
                "--openrouter-key",
                "test_openrouter_key",
                "--no-mcp",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["mcp_enabled"] is False

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_verbose_mode(self, mock_client_class):
        """Test send command with verbose output."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.id = "msg_128"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["send", "Hello world", "--api-key", "test_key", "--verbose"],
        )

        assert result.exit_code == 0
        assert "Sending message" in result.stdout
        assert "Message ID:" in result.stdout

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_handles_client_error(self, mock_client_class):
        """Test send command handles client errors gracefully."""
        # Setup mock to raise an error
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["send", "Hello world", "--api-key", "test_key"],
        )

        assert result.exit_code == 1
        assert "Error sending message:" in result.stdout


class TestGenerateLLMResponse:
    """Tests for the _generate_llm_response helper function."""

    @pytest.mark.asyncio
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_without_mcp(self, mock_llm_class):
        """Test LLM response generation without MCP tools."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        result = await _generate_llm_response(
            prompt="Test prompt",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            mcp_enabled=False,
        )

        assert result == "This is a test response"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_with_system_prompt(self, mock_llm_class):
        """Test LLM response generation with custom system prompt."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Expert response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        result = await _generate_llm_response(
            prompt="Analyze this",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            system_prompt="You are an expert analyst",
            mcp_enabled=False,
        )

        assert result == "Expert response"
        # Verify the system prompt was passed
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "You are an expert analyst"

    @pytest.mark.asyncio
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_strips_whitespace(self, mock_llm_class):
        """Test that response whitespace is stripped."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "  Response with whitespace  \n"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        result = await _generate_llm_response(
            prompt="Test",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            mcp_enabled=False,
        )

        assert result == "Response with whitespace"

    @pytest.mark.asyncio
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_handles_empty_content(self, mock_llm_class):
        """Test handling of empty response content."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        result = await _generate_llm_response(
            prompt="Test",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            mcp_enabled=False,
        )

        assert result == ""

    @pytest.mark.asyncio
    @patch("langchain.agents.create_agent")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_with_mcp(
        self, mock_llm_class, mock_mcp_client_class, mock_create_agent
    ):
        """Test LLM response generation with MCP tools."""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Setup mock MCP client
        mock_mcp_client = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_mcp_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_mcp_client_class.return_value = mock_mcp_client

        # Setup mock agent
        mock_agent = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Response from agent"
        # Simulate no tool_calls attribute for final message
        del mock_ai_message.tool_calls
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_ai_message]})
        mock_create_agent.return_value = mock_agent

        result = await _generate_llm_response(
            prompt="Test with MCP",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            mcp_enabled=True,
            mcp_server_url="https://test-mcp.example.com/sse",
        )

        assert result == "Response from agent"
        mock_mcp_client.get_tools.assert_called_once()
        mock_create_agent.assert_called_once()

    @pytest.mark.asyncio
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    @patch("langchain_openai.ChatOpenAI")
    async def test_generate_response_mcp_connection_error(
        self, mock_llm_class, mock_mcp_client_class
    ):
        """Test graceful handling when MCP connection fails."""
        # Setup mock LLM for fallback
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Fallback response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        # Simulate MCP connection error
        mock_mcp_client = MagicMock()
        mock_mcp_client.get_tools = AsyncMock(side_effect=Exception("Connection refused"))
        mock_mcp_client_class.return_value = mock_mcp_client

        result = await _generate_llm_response(
            prompt="Test",
            model_name="openai/gpt-4o",
            openrouter_api_key="test_key",
            mcp_enabled=True,
        )

        # Should fall back to direct LLM call
        assert result == "Fallback response"
        mock_llm.ainvoke.assert_called_once()


class TestSendCommandServerURL:
    """Tests for server URL handling in send command."""

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_converts_wss_to_https(self, mock_client_class):
        """Test that wss:// URLs are converted to https://."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello"
        mock_response.id = "msg_129"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        runner.invoke(
            app,
            [
                "send",
                "Hello",
                "--api-key",
                "test_key",
                "--server",
                "wss://custom.server.com",
            ],
        )

        # Check that AsyncTokenBowlClient was called with https URL
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["base_url"] == "https://custom.server.com"

    @patch("token_bowl_chat_agent.cli.AsyncTokenBowlClient")
    def test_send_converts_ws_to_http(self, mock_client_class):
        """Test that ws:// URLs are converted to http://."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Hello"
        mock_response.id = "msg_130"
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        runner.invoke(
            app,
            [
                "send",
                "Hello",
                "--api-key",
                "test_key",
                "--server",
                "ws://localhost:8080",
            ],
        )

        # Check that AsyncTokenBowlClient was called with http URL
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8080"
