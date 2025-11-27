"""Command-line interface for Token Bowl Chat Agent."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from token_bowl_chat import AsyncTokenBowlClient

from .agent import TokenBowlAgent

app = typer.Typer(
    name="token-bowl-agent",
    help="LangChain-powered intelligent agent for Token Bowl Chat servers.",
    no_args_is_help=True,
)
console = Console()

# Common options
ApiKey = Annotated[
    str | None,
    typer.Option(
        "--api-key",
        "-k",
        envvar="TOKEN_BOWL_CHAT_API_KEY",
        help="Token Bowl Chat API key",
    ),
]
OpenRouterKey = Annotated[
    str | None,
    typer.Option(
        "--openrouter-key",
        "-o",
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key",
    ),
]
SystemPrompt = Annotated[
    str | None,
    typer.Option(
        "--system",
        "-s",
        help="System prompt (agent personality) - text or path to markdown file",
    ),
]
UserPrompt = Annotated[
    str | None,
    typer.Option(
        "--user",
        "-u",
        help="User prompt (batch processing instructions) - text or file path",
    ),
]
Model = Annotated[
    str,
    typer.Option(
        "--model",
        "-m",
        help="OpenRouter model name",
    ),
]
Server = Annotated[
    str,
    typer.Option(
        "--server",
        help="WebSocket server URL",
    ),
]
QueueInterval = Annotated[
    float,
    typer.Option(
        "--queue-interval",
        "-q",
        help="Seconds to wait before flushing message queue",
    ),
]
MaxReconnectDelay = Annotated[
    float,
    typer.Option(
        "--max-reconnect-delay",
        help="Maximum delay between reconnection attempts (seconds)",
    ),
]
ContextWindow = Annotated[
    int,
    typer.Option(
        "--context-window",
        "-c",
        help="Maximum context window in tokens for conversation history",
    ),
]
CooldownMessages = Annotated[
    int,
    typer.Option(
        "--cooldown-messages",
        help="Number of messages before cooldown starts",
    ),
]
CooldownMinutes = Annotated[
    int,
    typer.Option(
        "--cooldown-minutes",
        help="Cooldown duration in minutes",
    ),
]
MaxConversationHistory = Annotated[
    int,
    typer.Option(
        "--max-conversation-history",
        help="Maximum number of messages to keep in conversation history",
    ),
]
MaxRetryAttempts = Annotated[
    int,
    typer.Option(
        "--max-retry-attempts",
        help="Maximum number of retry attempts per failed message",
    ),
]
RetryBaseDelay = Annotated[
    float,
    typer.Option(
        "--retry-base-delay",
        help="Base delay in seconds for exponential backoff",
    ),
]
MaxRetryDelay = Annotated[
    float,
    typer.Option(
        "--max-retry-delay",
        help="Maximum retry delay in seconds",
    ),
]
McpEnabled = Annotated[
    bool,
    typer.Option(
        "--mcp/--no-mcp",
        help="Enable/disable MCP (Model Context Protocol) tools",
    ),
]
McpServer = Annotated[
    str,
    typer.Option(
        "--mcp-server",
        help="MCP server URL (SSE transport)",
    ),
]
Verbose = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
]


@app.command()
def run(
    api_key: ApiKey = None,
    openrouter_key: OpenRouterKey = None,
    system: SystemPrompt = None,
    user: UserPrompt = None,
    model: Model = "openai/gpt-4o-mini",
    server: Server = "wss://api.tokenbowl.ai",
    queue_interval: QueueInterval = 15.0,
    max_reconnect_delay: MaxReconnectDelay = 300.0,
    context_window: ContextWindow = 128000,
    cooldown_messages: CooldownMessages = 3,
    cooldown_minutes: CooldownMinutes = 10,
    max_conversation_history: MaxConversationHistory = 10,
    max_retry_attempts: MaxRetryAttempts = 3,
    retry_base_delay: RetryBaseDelay = 5.0,
    max_retry_delay: MaxRetryDelay = 60.0,
    mcp: McpEnabled = True,
    mcp_server: McpServer = "https://tokenbowl-mcp.haihai.ai/sse",
    verbose: Verbose = False,
) -> None:
    """Run the agent continuously, responding to inbound messages."""
    if not api_key:
        console.print(
            "[red]Error: Token Bowl Chat API key required. "
            "Set TOKEN_BOWL_CHAT_API_KEY or use --api-key[/red]"
        )
        raise typer.Exit(1)

    if not openrouter_key:
        console.print(
            "[red]Error: OpenRouter API key required. "
            "Set OPENROUTER_API_KEY or use --openrouter-key[/red]"
        )
        raise typer.Exit(1)

    agent = TokenBowlAgent(
        api_key=api_key,
        openrouter_api_key=openrouter_key,
        system_prompt=system,
        user_prompt=user,
        model_name=model,
        server_url=server,
        queue_interval=queue_interval,
        max_reconnect_delay=max_reconnect_delay,
        context_window=context_window,
        cooldown_messages=cooldown_messages,
        cooldown_minutes=cooldown_minutes,
        max_conversation_history=max_conversation_history,
        mcp_enabled=mcp,
        mcp_server_url=mcp_server,
        max_retry_attempts=max_retry_attempts,
        retry_base_delay=retry_base_delay,
        max_retry_delay=max_retry_delay,
        verbose=verbose,
    )

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent stopped by user[/yellow]")


@app.command()
def send(
    message: Annotated[
        str, typer.Argument(help="Message to send (or prompt for LLM if --model provided)")
    ],
    api_key: ApiKey = None,
    to: Annotated[
        str | None,
        typer.Option(
            "--to",
            "-t",
            help="Recipient username for direct message (omit for room message)",
        ),
    ] = None,
    server: Server = "wss://api.tokenbowl.ai",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="OpenRouter model name (enables LLM processing)",
        ),
    ] = None,
    openrouter_key: OpenRouterKey = None,
    system: SystemPrompt = None,
    mcp: McpEnabled = True,
    mcp_server: McpServer = "https://tokenbowl-mcp.haihai.ai/sse",
    verbose: Verbose = False,
) -> None:
    """Send a single message to the chat server.

    If --model is provided, the message is treated as a prompt and processed
    through the LLM (with optional MCP tools) before sending the response.
    """
    if not api_key:
        console.print(
            "[red]Error: Token Bowl Chat API key required. "
            "Set TOKEN_BOWL_CHAT_API_KEY or use --api-key[/red]"
        )
        raise typer.Exit(1)

    if model and not openrouter_key:
        console.print(
            "[red]Error: OpenRouter API key required when using --model. "
            "Set OPENROUTER_API_KEY or use --openrouter-key[/red]"
        )
        raise typer.Exit(1)

    async def _send() -> None:
        # Convert WebSocket URL to HTTP URL for the REST API
        http_url = server.replace("wss://", "https://").replace("ws://", "http://")

        # Determine the content to send
        content_to_send = message

        # If model is provided, process through LLM first
        if model:
            content_to_send = await _generate_llm_response(
                prompt=message,
                model_name=model,
                openrouter_api_key=openrouter_key,  # type: ignore[arg-type]
                system_prompt=system,
                mcp_enabled=mcp,
                mcp_server_url=mcp_server,
                verbose=verbose,
            )

        async with AsyncTokenBowlClient(api_key=api_key, base_url=http_url) as client:
            if verbose:
                target = f"to @{to}" if to else "to room"
                console.print(f"[dim]Sending message {target}...[/dim]")

            response = await client.send_message(content_to_send, to_username=to)

            if to:
                console.print(f"[green]✓ Sent DM to @{to}:[/green] {response.content[:100]}")
            else:
                console.print(f"[green]✓ Sent to room:[/green] {response.content[:100]}")

            if verbose:
                console.print(f"[dim]Message ID: {response.id}[/dim]")

    try:
        asyncio.run(_send())
    except Exception as e:
        console.print(f"[red]Error sending message: {e}[/red]")
        raise typer.Exit(1) from None


async def _generate_llm_response(
    prompt: str,
    model_name: str,
    openrouter_api_key: str,
    system_prompt: str | None = None,
    mcp_enabled: bool = True,
    mcp_server_url: str = "https://tokenbowl-mcp.haihai.ai/sse",
    verbose: bool = False,
) -> str:
    """Generate a response using LLM with optional MCP tools.

    Args:
        prompt: The user prompt to process
        model_name: OpenRouter model name
        openrouter_api_key: OpenRouter API key
        system_prompt: Optional system prompt (text or file path)
        mcp_enabled: Whether to enable MCP tools
        mcp_server_url: MCP server URL
        verbose: Enable verbose output

    Returns:
        Generated response text
    """
    from pathlib import Path

    from langchain.agents import create_agent
    from langchain_community.callbacks import get_openai_callback
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    # Load system prompt from file if it's a path
    resolved_system_prompt = "You are a helpful assistant."
    if system_prompt:
        path = Path(system_prompt)
        if path.exists() and path.is_file():
            try:
                resolved_system_prompt = path.read_text(encoding="utf-8")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read prompt file: {e}[/yellow]")
        else:
            resolved_system_prompt = system_prompt

    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        api_key=SecretStr(openrouter_api_key),
        base_url="https://openrouter.ai/api/v1",
        streaming=False,
        default_headers={
            "HTTP-Referer": "https://github.com/RobSpectre/token-bowl-chat",
            "X-Title": "Token Bowl Chat Agent",
        },
    )

    if verbose:
        console.print(f"[dim]Initialized LLM: {model_name}[/dim]")

    # Initialize MCP if enabled
    mcp_tools: list = []
    agent = None

    if mcp_enabled:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            mcp_client = MultiServerMCPClient(
                {
                    "tokenbowl": {
                        "transport": "sse",
                        "url": mcp_server_url,
                    }
                }
            )

            mcp_tools = await mcp_client.get_tools()

            if verbose:
                console.print(f"[dim]MCP: Connected with {len(mcp_tools)} tools[/dim]")

            if mcp_tools:
                agent = create_agent(
                    llm,
                    tools=mcp_tools,
                    system_prompt=resolved_system_prompt,
                )

        except ImportError:
            if verbose:
                console.print("[dim]MCP libraries not available, skipping[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"[yellow]MCP initialization failed: {e}[/yellow]")

    # Generate response
    with get_openai_callback() as cb:
        if agent:
            # Use agent with tools
            result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})

            # Extract response from the last AI message
            # Look for a message with content but no active tool calls
            response_text = ""
            if "messages" in result:
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and msg.content:
                        # Skip messages that have tool_calls (these are tool invocations)
                        tool_calls = getattr(msg, "tool_calls", None)
                        if not tool_calls:
                            response_text = str(msg.content)
                            break
        else:
            # Direct LLM call without tools
            messages = [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = await llm.ainvoke(messages)
            response_text = str(response.content) if response.content else ""

        if verbose:
            console.print(
                f"[dim]Tokens used: {cb.total_tokens} "
                f"({cb.prompt_tokens} in, {cb.completion_tokens} out)[/dim]"
            )

    return response_text.strip()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
