#!/usr/bin/env python3
"""Basic example of using the Token Bowl Chat Agent."""

import asyncio
import os

from token_bowl_chat_agent import TokenBowlAgent


async def main():
    """Run the basic agent example."""
    # Get API keys from environment
    api_key = os.getenv("TOKEN_BOWL_CHAT_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or not openrouter_key:
        print("Please set TOKEN_BOWL_CHAT_API_KEY and OPENROUTER_API_KEY environment variables")
        return

    # Create the agent with custom configuration
    agent = TokenBowlAgent(
        api_key=api_key,
        openrouter_api_key=openrouter_key,
        model_name="openai/gpt-4o-mini",  # Use a cost-effective model
        system_prompt="You are a helpful assistant for a fantasy football league",
        user_prompt="Please help league members with their fantasy football questions",
        queue_interval=30.0,  # Process messages every 30 seconds
        max_conversation_history=10,  # Keep last 10 messages in memory
        cooldown_messages=3,  # Cooldown after 3 messages
        cooldown_minutes=10,  # 10 minute cooldown period
        verbose=True,  # Enable verbose logging
    )

    # Run the agent
    print("Starting Token Bowl Chat Agent...")
    print("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\n\nAgent stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
