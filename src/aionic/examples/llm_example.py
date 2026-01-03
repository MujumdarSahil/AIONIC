"""
LLM Integration Example - Demonstrates Multi-LLM usage in AIONIC.

This example shows how to:
1. Set up LLM providers (OpenAI, Claude, Gemini, Groq)
2. Configure LLM router with strategies
3. Use LLM in agents for text generation, chat, and embeddings
4. Handle failover and routing decisions
"""

import os
from typing import Dict, Any

from aionic.core.orchestrator import Orchestrator
from aionic.core.task import TaskPriority
from aionic.core.tool import ToolRegistry
from aionic.memory.memory_store import InMemoryMemoryStore
from aionic.memory.audit_logger import InMemoryAuditLogger
from aionic.security.autonomy_policy import AutonomyPolicy
from aionic.agents.base_agent import BaseAgent

# LLM Integration
from aionic.llm.registry import ProviderRegistry
from aionic.llm.router import LLMRouter, RoutingStrategy
from aionic.llm.models import ModelConfig, ChatMessage, MessageRole
from aionic.llm.providers import (
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    GroqProvider,
)


def setup_llm_system() -> LLMRouter:
    """
    Set up LLM providers and router.
    
    This demonstrates how to:
    - Register multiple providers
    - Configure routing strategies
    - Set up failover behavior
    """
    # Create provider registry
    registry = ProviderRegistry()
    
    # Register providers (API keys from environment variables)
    # Note: In production, use secure credential management
    
    # OpenAI
    try:
        openai_provider = OpenAIProvider()
        openai_provider.initialize()
        registry.register(openai_provider)
        print("✓ OpenAI provider registered")
    except Exception as e:
        print(f"✗ OpenAI provider failed: {e}")
    
    # Claude
    try:
        claude_provider = ClaudeProvider()
        claude_provider.initialize()
        registry.register(claude_provider)
        print("✓ Claude provider registered")
    except Exception as e:
        print(f"✗ Claude provider failed: {e}")
    
    # Gemini
    try:
        gemini_provider = GeminiProvider()
        gemini_provider.initialize()
        registry.register(gemini_provider)
        print("✓ Gemini provider registered")
    except Exception as e:
        print(f"✗ Gemini provider failed: {e}")
    
    # Groq (fast inference)
    try:
        groq_provider = GroqProvider()
        groq_provider.initialize()
        registry.register(groq_provider)
        print("✓ Groq provider registered")
    except Exception as e:
        print(f"✗ Groq provider failed: {e}")
    
    # Create audit logger for routing decisions
    audit_logger = InMemoryAuditLogger()
    
    # Create router with quality-optimized strategy
    router = LLMRouter(
        registry=registry,
        audit_logger=audit_logger,
        default_strategy=RoutingStrategy.QUALITY_OPTIMIZED,
    )
    
    # Configure provider preferences for different strategies
    router.set_provider_preferences(
        RoutingStrategy.COST_OPTIMIZED,
        ["groq", "openai", "gemini"],  # Groq is cheapest
    )
    
    router.set_provider_preferences(
        RoutingStrategy.SPEED_OPTIMIZED,
        ["groq", "openai", "claude"],  # Groq is fastest
    )
    
    router.set_provider_preferences(
        RoutingStrategy.QUALITY_OPTIMIZED,
        ["claude", "openai", "gemini"],  # Claude/OpenAI for quality
    )
    
    print(f"\n✓ LLM Router configured with {len(registry.list_providers())} providers")
    return router


def example_basic_generation(router: LLMRouter):
    """Example: Basic text generation."""
    print("\n" + "="*60)
    print("Example 1: Basic Text Generation")
    print("="*60)
    
    prompt = "Explain quantum computing in one sentence."
    
    try:
        response = router.generate(
            prompt=prompt,
            config=ModelConfig(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100,
            ),
        )
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response.content}")
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_chat_conversation(router: LLMRouter):
    """Example: Chat conversation with context."""
    print("\n" + "="*60)
    print("Example 2: Chat Conversation")
    print("="*60)
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]
    
    try:
        response = router.chat(
            messages=messages,
            config=ModelConfig(model="gpt-3.5-turbo"),
            strategy=RoutingStrategy.SPEED_OPTIMIZED,  # Use fast provider
        )
        
        print("Conversation:")
        for msg in messages:
            print(f"  {msg.role.value}: {msg.content}")
        print(f"  assistant: {response.content}")
        print(f"\nProvider: {response.provider}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_agent_with_llm():
    """Example: Agent using LLM for reasoning."""
    print("\n" + "="*60)
    print("Example 3: Agent with LLM Integration")
    print("="*60)
    
    # Setup
    memory_store = InMemoryMemoryStore()
    audit_logger = InMemoryAuditLogger()
    autonomy_policy = AutonomyPolicy()
    tool_registry = ToolRegistry()
    
    # Create LLM router
    llm_router = setup_llm_system()
    
    # Create agent with LLM router
    agent = BaseAgent(
        agent_id="llm_agent_1",
        name="LLM-Enabled Agent",
        goal="Answer questions using LLM capabilities",
        memory=memory_store,
        autonomy_policy=autonomy_policy,
        tool_registry=tool_registry,
        llm_router=llm_router,
    )
    
    # Agent uses LLM to generate text
    try:
        prompt = "Write a haiku about artificial intelligence."
        response = agent.generate_text(
            prompt=prompt,
            config=ModelConfig(temperature=0.9, max_tokens=50),
        )
        
        print(f"Agent: {agent.name}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Check reasoning log
        summary = agent.get_reasoning_summary()
        print(f"\nAgent reasoning entries: {summary['reasoning_entries']}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_embeddings(router: LLMRouter):
    """Example: Generate embeddings."""
    print("\n" + "="*60)
    print("Example 4: Text Embeddings")
    print("="*60)
    
    text = "The quick brown fox jumps over the lazy dog."
    
    try:
        # Note: Not all providers support embeddings
        # OpenAI and Gemini do, Claude and Groq do not
        response = router.embed(
            text=text,
            config=ModelConfig(model="text-embedding-ada-002"),
        )
        
        print(f"Text: {text}")
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Embedding dimension: {len(response.embeddings[0]) if response.embeddings else 0}")
        print(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_routing_strategies(router: LLMRouter):
    """Example: Different routing strategies."""
    print("\n" + "="*60)
    print("Example 5: Routing Strategies")
    print("="*60)
    
    prompt = "What is machine learning?"
    
    strategies = [
        RoutingStrategy.COST_OPTIMIZED,
        RoutingStrategy.SPEED_OPTIMIZED,
        RoutingStrategy.QUALITY_OPTIMIZED,
    ]
    
    for strategy in strategies:
        try:
            response = router.generate(
                prompt=prompt,
                strategy=strategy,
                config=ModelConfig(max_tokens=50),
            )
            
            print(f"\nStrategy: {strategy.value}")
            print(f"  Provider: {response.provider}")
            print(f"  Model: {response.model}")
            print(f"  Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"\nStrategy: {strategy.value}")
            print(f"  Error: {e}")


def example_failover():
    """Example: Automatic failover on provider failure."""
    print("\n" + "="*60)
    print("Example 6: Failover Behavior")
    print("="*60)
    
    router = setup_llm_system()
    
    # Simulate a scenario where primary provider might fail
    # Router will automatically try next available provider
    prompt = "Explain failover in distributed systems."
    
    try:
        response = router.generate(
            prompt=prompt,
            strategy=RoutingStrategy.PROVIDER_FALLBACK,
            max_retries=3,  # Try up to 3 providers
        )
        
        print(f"Prompt: {prompt}")
        print(f"Final provider: {response.provider}")
        print(f"Response: {response.content[:200]}...")
        
        # Show routing stats
        stats = router.get_routing_stats()
        print(f"\nRouting Statistics:")
        print(f"  Available providers: {stats['available_providers']}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("AIONIC Multi-LLM Integration Examples")
    print("="*60)
    print("\nNote: This example requires API keys in environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY (for Claude)")
    print("  - GOOGLE_API_KEY (for Gemini)")
    print("  - GROQ_API_KEY (for Groq)")
    print("\nAt least one provider must be configured for examples to work.")
    
    # Setup LLM system
    router = setup_llm_system()
    
    if not router.registry.get_available_providers():
        print("\n⚠️  No LLM providers available. Please configure at least one API key.")
        return
    
    # Run examples
    try:
        example_basic_generation(router)
        example_chat_conversation(router)
        example_agent_with_llm()
        example_embeddings(router)
        example_routing_strategies(router)
        example_failover()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")


if __name__ == "__main__":
    main()

