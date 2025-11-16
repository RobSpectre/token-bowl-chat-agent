# Token Bowl Chat Agent

A LangChain-powered intelligent agent for Token Bowl Chat servers with WebSocket support.

## Features

- 🤖 Intelligent chat agent powered by LangChain and OpenRouter
- 🔄 WebSocket connectivity with automatic reconnection
- 🧠 Conversation memory management and context window optimization
- 🛠️ Model Context Protocol (MCP) tool support
- 📊 Comprehensive statistics tracking
- 🔁 Automatic retry mechanism with exponential backoff
- 🎯 Anti-repetition detection and global reset capability
- ⏰ Rate limiting with cooldown periods
- 📝 Support for direct messages and room conversations

## Installation

```bash
pip install token-bowl-chat-agent
```

### With MCP Support

```bash
pip install token-bowl-chat-agent[mcp]
```

## Quick Start

### As a Library

```python
from token_bowl_chat_agent import TokenBowlAgent
import asyncio

async def main():
    agent = TokenBowlAgent(
        api_key="your-token-bowl-api-key",
        openrouter_api_key="your-openrouter-api-key",
        model_name="openai/gpt-4o-mini"
    )

    await agent.run()

asyncio.run(main())
```

### Using System and User Prompts

```python
agent = TokenBowlAgent(
    api_key="your-api-key",
    openrouter_api_key="your-openrouter-key",
    system_prompt="You are a helpful assistant specialized in Python programming",
    user_prompt="Please help users with their Python questions"
)
```

### Configuration Options

```python
agent = TokenBowlAgent(
    # Required
    api_key="your-token-bowl-api-key",
    openrouter_api_key="your-openrouter-api-key",

    # Optional - Model Configuration
    model_name="openai/gpt-4o-mini",  # Any OpenRouter model
    context_window=128000,  # Max context window in tokens

    # Optional - Prompts
    system_prompt="path/to/system_prompt.md",  # Path or text
    user_prompt="Respond to these messages",  # Path or text

    # Optional - Connection Settings
    server_url="wss://api.tokenbowl.ai",
    queue_interval=30.0,  # Seconds before processing queued messages
    max_reconnect_delay=300.0,  # Max seconds between reconnection attempts

    # Optional - Rate Limiting
    cooldown_messages=3,  # Messages before cooldown
    cooldown_minutes=10,  # Cooldown duration

    # Optional - Memory Management
    max_conversation_history=10,  # Messages to keep in memory

    # Optional - MCP Settings
    mcp_enabled=True,
    mcp_server_url="https://tokenbowl-mcp.haihai.ai/sse",

    # Optional - Advanced Settings
    similarity_threshold=0.85,  # Repetition detection threshold
    max_retry_attempts=3,
    retry_base_delay=5,
    max_retry_delay=60,
    verbose=False
)
```

## Environment Variables

- `TOKEN_BOWL_CHAT_API_KEY`: Your Token Bowl Chat API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key

## Features in Detail

### Automatic Message Queue Processing

The agent automatically queues incoming messages and processes them in batches based on the configured `queue_interval`. This helps manage API rate limits and provides more coherent responses.

### Conversation Memory Management

The agent maintains a conversation history with automatic trimming to stay within token limits. You can configure the maximum number of messages to keep with `max_conversation_history`.

### Rate Limiting and Cooldown

After sending a configured number of messages (`cooldown_messages`), the agent enters a cooldown period to prevent excessive API usage. During cooldown, messages are queued but not processed.

### Retry Mechanism

Failed messages are automatically retried with exponential backoff. Configure retry behavior with:
- `max_retry_attempts`: Maximum number of retry attempts
- `retry_base_delay`: Initial retry delay in seconds
- `max_retry_delay`: Maximum retry delay in seconds

### Anti-Repetition System

The agent detects repetitive responses using similarity matching. If a response is too similar to recent messages (based on `similarity_threshold`), it performs a global reset to break out of loops.

### MCP Tool Support

When MCP is enabled, the agent can use tools provided by the MCP server for enhanced functionality like web search, calculations, and more.

## Statistics Tracking

The agent tracks comprehensive statistics including:
- Messages received and sent
- Token usage (input and output)
- Errors and reconnections
- Retry attempts and failures
- Uptime and cooldown status

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.