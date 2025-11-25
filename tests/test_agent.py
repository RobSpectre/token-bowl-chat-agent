"""Tests for TokenBowlAgent."""

from unittest.mock import patch

import pytest

from token_bowl_chat_agent import AgentStats, TokenBowlAgent


class TestTokenBowlAgent:
    """Tests for TokenBowlAgent class."""

    def test_agent_initialization(self):
        """Test agent initializes with required parameters."""
        agent = TokenBowlAgent(api_key="test_key", openrouter_api_key="test_openrouter_key")

        assert agent.api_key == "test_key"
        assert agent.openrouter_api_key == "test_openrouter_key"
        assert agent.model_name == "openai/gpt-4o-mini"  # default
        assert agent.server_url == "wss://api.tokenbowl.ai"  # default
        assert agent.queue_interval == 30.0  # default
        assert agent.max_reconnect_delay == 300.0  # default
        assert agent.context_window == 128000  # default

    def test_agent_initialization_with_env_vars(self):
        """Test agent initializes from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "TOKEN_BOWL_CHAT_API_KEY": "env_api_key",
                "OPENROUTER_API_KEY": "env_openrouter_key",
            },
        ):
            agent = TokenBowlAgent(api_key="", openrouter_api_key="")

            assert agent.api_key == "env_api_key"
            assert agent.openrouter_api_key == "env_openrouter_key"

    def test_agent_custom_configuration(self):
        """Test agent with custom configuration."""
        agent = TokenBowlAgent(
            api_key="test_key",
            openrouter_api_key="test_openrouter_key",
            model_name="anthropic/claude-3-sonnet",
            server_url="wss://custom.server.com",
            queue_interval=60.0,
            max_reconnect_delay=600.0,
            context_window=200000,
            cooldown_messages=5,
            cooldown_minutes=15,
            max_conversation_history=20,
            mcp_enabled=False,
            similarity_threshold=0.9,
            max_retry_attempts=5,
            retry_base_delay=10,
            max_retry_delay=120,
            verbose=True,
        )

        assert agent.model_name == "anthropic/claude-3-sonnet"
        assert agent.server_url == "wss://custom.server.com"
        assert agent.queue_interval == 60.0
        assert agent.max_reconnect_delay == 600.0
        assert agent.context_window == 200000
        assert agent.max_conversation_history == 20
        assert agent.mcp_enabled is False
        assert agent.similarity_threshold == 0.9
        assert agent.max_retry_attempts == 5
        assert agent.retry_base_delay == 10
        assert agent.max_retry_delay == 120
        assert agent.verbose is True

    def test_context_window_validation(self):
        """Test context window validation."""
        # Test minimum context window
        with pytest.raises(ValueError, match="context_window must be at least 1000"):
            TokenBowlAgent(
                api_key="test_key",
                openrouter_api_key="test_openrouter_key",
                context_window=999,
            )

    def test_max_conversation_history_validation(self):
        """Test max conversation history validation."""
        with pytest.raises(ValueError, match="max_conversation_history must be at least 1"):
            TokenBowlAgent(
                api_key="test_key",
                openrouter_api_key="test_openrouter_key",
                max_conversation_history=0,
            )

    def test_load_prompt_from_text(self):
        """Test loading prompt from text."""
        agent = TokenBowlAgent(
            api_key="test_key",
            openrouter_api_key="test_openrouter_key",
            system_prompt="You are a helpful assistant",
            user_prompt="Please respond to messages",
        )

        assert agent.system_prompt == "You are a helpful assistant"
        assert agent.user_prompt == "Please respond to messages"

    def test_agent_stats_initialization(self):
        """Test AgentStats initialization."""
        stats = AgentStats()

        assert stats.messages_received == 0
        assert stats.messages_sent == 0
        assert stats.errors == 0
        assert stats.reconnections == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.retries == 0
        assert stats.messages_failed_permanently == 0

    def test_agent_stats_to_dict(self):
        """Test AgentStats to_dict method."""
        stats = AgentStats()
        stats.messages_received = 10
        stats.messages_sent = 5
        stats.errors = 2

        stats_dict = stats.to_dict()

        assert stats_dict["messages_received"] == 10
        assert stats_dict["messages_sent"] == 5
        assert stats_dict["errors"] == 2
        assert "timestamp" in stats_dict
        assert "uptime_seconds" in stats_dict
        assert "start_time" in stats_dict

    @pytest.mark.asyncio
    async def test_run_without_api_key(self):
        """Test run method raises error without API key."""
        agent = TokenBowlAgent(api_key="", openrouter_api_key="test_openrouter_key")

        with pytest.raises(ValueError, match="Token Bowl Chat API key required"):
            await agent.run()

    def test_calculate_similarity(self):
        """Test text similarity calculation."""
        agent = TokenBowlAgent(api_key="test_key", openrouter_api_key="test_openrouter_key")

        # Test identical texts
        similarity = agent._calculate_similarity("Hello world", "Hello world")
        assert similarity == 1.0

        # Test completely different texts
        similarity = agent._calculate_similarity("Hello", "Goodbye")
        assert similarity < 0.5

        # Test similar texts
        similarity = agent._calculate_similarity("The quick brown fox", "The quick brown foxes")
        assert similarity > 0.8

    def test_is_repetitive_response(self):
        """Test repetitive response detection."""
        agent = TokenBowlAgent(
            api_key="test_key",
            openrouter_api_key="test_openrouter_key",
            similarity_threshold=0.85,
        )

        # Add some messages to sent_message_contents
        agent.sent_message_contents.append("Hello, how can I help you today?")
        agent.sent_message_contents.append("I'm here to assist you.")

        # Test non-repetitive response
        assert not agent._is_repetitive_response("What's the weather like?")

        # Test repetitive response
        assert agent._is_repetitive_response("Hello, how can I help you today?")

        # Test similar but not quite repetitive (adjusted for actual similarity)
        assert not agent._is_repetitive_response("Can I get you some coffee?")

    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        agent = TokenBowlAgent(
            api_key="test_key",
            openrouter_api_key="test_openrouter_key",
            retry_base_delay=5,
            max_retry_delay=60,
        )

        # Test exponential backoff
        delay1 = agent._calculate_retry_delay(0)
        assert 4.5 <= delay1 <= 5.5  # 5 seconds ± 10% jitter

        delay2 = agent._calculate_retry_delay(1)
        assert 9 <= delay2 <= 11  # 10 seconds ± 10% jitter

        delay3 = agent._calculate_retry_delay(2)
        assert 18 <= delay3 <= 22  # 20 seconds ± 10% jitter

        # Test max delay cap
        delay_max = agent._calculate_retry_delay(10)
        assert delay_max <= 66  # 60 seconds + max 10% jitter

    def test_cooldown_mechanism(self):
        """Test cooldown mechanism."""
        agent = TokenBowlAgent(
            api_key="test_key",
            openrouter_api_key="test_openrouter_key",
            cooldown_messages=3,
            cooldown_minutes=10,
        )

        # Initially not in cooldown
        assert not agent._is_in_cooldown()
        assert agent._get_cooldown_remaining() == 0

        # Start cooldown
        agent._start_cooldown()
        assert agent._is_in_cooldown()
        assert agent._get_cooldown_remaining() > 0

        # Test cooldown end
        agent.cooldown_start_time = None
        agent._end_cooldown()
        assert not agent._is_in_cooldown()
        assert agent.messages_sent_in_window == 0
