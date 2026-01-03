"""
Unified LLM Configuration - Single-file configuration for all LLM providers.

Provides a centralized configuration system that loads all provider
configurations from environment variables and allows runtime overrides.

Supports:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Claude (Anthropic)
- Gemini (Google)
- Groq
- Local/Ollama models
- vLLM

Design:
- Environment variables loaded from .env by default
- Safe defaults with missing-key warnings
- Runtime configuration overrides
- Transparent integration with router
"""

import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

from .registry import ProviderRegistry
from .router import LLMRouter, RoutingStrategy
from .models import ModelConfig
from .providers import (
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    GroqProvider,
    LocalModelProvider,
)


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key: Optional[str] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)


class LLMConfigLoader:
    """
    Unified loader for LLM provider configurations.
    
    Loads all provider configurations from environment variables
    and provides a simple interface to create providers and routers.
    """
    
    # Model name mappings to provider/model pairs
    MODEL_MAPPINGS: Dict[str, Dict[str, str]] = {
        # OpenAI models
        "gpt-4": {"provider": "openai", "model": "gpt-4"},
        "gpt-4-turbo": {"provider": "openai", "model": "gpt-4-turbo-preview"},
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-3.5": {"provider": "openai", "model": "gpt-3.5-turbo"},
        
        # Claude models
        "claude-3-opus": {"provider": "claude", "model": "claude-3-opus-20240229"},
        "claude-3-sonnet": {"provider": "claude", "model": "claude-3-sonnet-20240229"},
        "claude-3-haiku": {"provider": "claude", "model": "claude-3-haiku-20240307"},
        "claude-3": {"provider": "claude", "model": "claude-3-sonnet-20240229"},
        
        # Gemini models
        "gemini-pro": {"provider": "gemini", "model": "gemini-pro"},
        "gemini-ultra": {"provider": "gemini", "model": "gemini-ultra"},
        "gemini": {"provider": "gemini", "model": "gemini-pro"},
        
        # Groq models
        "groq-mixtral": {"provider": "groq", "model": "mixtral-8x7b-32768"},
        "groq-llama": {"provider": "groq", "model": "llama2-70b-4096"},
        "groq": {"provider": "groq", "model": "mixtral-8x7b-32768"},
        
        # Local models
        "local-llama": {"provider": "local", "model": "llama2"},
        "local-mistral": {"provider": "local", "model": "mistral"},
        "local": {"provider": "local", "model": "llama2"},
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            env_file: Optional path to .env file (defaults to .env in current directory)
        """
        self.env_file = env_file or ".env"
        self._load_env_file()
        self._provider_configs: Dict[str, LLMProviderConfig] = {}
        self._load_provider_configs()
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        if os.path.exists(self.env_file):
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
            except ImportError:
                # dotenv not installed, skip
                pass
    
    def _load_provider_configs(self) -> None:
        """Load all provider configurations from environment."""
        # OpenAI
        self._provider_configs["openai"] = LLMProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            enabled=os.getenv("OPENAI_ENABLED", "true").lower() == "true",
            config={
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "organization": os.getenv("OPENAI_ORG_ID"),
            },
        )
        
        # Claude
        self._provider_configs["claude"] = LLMProviderConfig(
            name="claude",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            enabled=os.getenv("CLAUDE_ENABLED", "true").lower() == "true",
            config={},
        )
        
        # Gemini
        self._provider_configs["gemini"] = LLMProviderConfig(
            name="gemini",
            api_key=os.getenv("GEMINI_API_KEY"),
            enabled=os.getenv("GEMINI_ENABLED", "true").lower() == "true",
            config={},
        )
        
        # Groq
        self._provider_configs["groq"] = LLMProviderConfig(
            name="groq",
            api_key=os.getenv("GROQ_API_KEY"),
            enabled=os.getenv("GROQ_ENABLED", "true").lower() == "true",
            config={},
        )
        
        # Local/Ollama
        self._provider_configs["local"] = LLMProviderConfig(
            name="local",
            api_key=None,  # Local models don't need API keys
            enabled=os.getenv("LOCAL_ENABLED", "true").lower() == "true",
            config={
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "model": os.getenv("LOCAL_MODEL", "llama2"),
            },
        )
    
    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a provider."""
        return self._provider_configs.get(provider_name.lower())
    
    def create_provider(
        self,
        provider_name: str,
        api_key_override: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a provider instance from configuration.
        
        Args:
            provider_name: Provider name (openai, claude, gemini, groq, local)
            api_key_override: Optional API key override
            config_override: Optional config override
            
        Returns:
            Provider instance or None if not available
        """
        provider_name = provider_name.lower()
        config = self._provider_configs.get(provider_name)
        
        if not config or not config.enabled:
            return None
        
        api_key = api_key_override or config.api_key
        merged_config = {**config.config, **(config_override or {})}
        
        # Create provider instance
        provider_map = {
            "openai": OpenAIProvider,
            "claude": ClaudeProvider,
            "gemini": GeminiProvider,
            "groq": GroqProvider,
            "local": LocalModelProvider,
        }
        
        provider_class = provider_map.get(provider_name)
        if not provider_class:
            return None
        
        try:
            provider = provider_class(api_key=api_key, config=merged_config)
            provider.initialize()
            return provider
        except Exception as e:
            print(f"Warning: Failed to initialize {provider_name} provider: {e}")
            return None
    
    def create_registry(
        self,
        auto_register: bool = True,
        provider_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ProviderRegistry:
        """
        Create and populate a provider registry.
        
        Args:
            auto_register: Whether to automatically register all enabled providers
            provider_overrides: Optional overrides for specific providers
            
        Returns:
            Populated ProviderRegistry
        """
        registry = ProviderRegistry()
        
        if auto_register:
            provider_overrides = provider_overrides or {}
            
            for provider_name, config in self._provider_configs.items():
                if not config.enabled:
                    continue
                
                override = provider_overrides.get(provider_name, {})
                provider = self.create_provider(
                    provider_name,
                    api_key_override=override.get("api_key"),
                    config_override=override.get("config"),
                )
                
                if provider:
                    try:
                        registry.register(provider)
                    except Exception as e:
                        print(f"Warning: Failed to register {provider_name}: {e}")
        
        return registry
    
    def create_router(
        self,
        registry: Optional[ProviderRegistry] = None,
        default_strategy: RoutingStrategy = RoutingStrategy.QUALITY_OPTIMIZED,
        auto_register_providers: bool = True,
    ) -> LLMRouter:
        """
        Create an LLM router with configured providers.
        
        Args:
            registry: Optional existing registry (creates new one if None)
            default_strategy: Default routing strategy
            auto_register_providers: Whether to auto-register providers
            
        Returns:
            Configured LLMRouter
        """
        if registry is None:
            registry = self.create_registry(auto_register=auto_register_providers)
        
        return LLMRouter(
            registry=registry,
            default_strategy=default_strategy,
        )
    
    def get_model_config(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Optional[ModelConfig]:
        """
        Get ModelConfig for a model by friendly name.
        
        Args:
            model_name: Friendly model name (e.g., "gpt-4", "claude-3", "gemini-pro")
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional config parameters
            
        Returns:
            ModelConfig or None if model not found
        """
        mapping = self.MODEL_MAPPINGS.get(model_name.lower())
        if not mapping:
            return None
        
        return ModelConfig(
            model=mapping["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    def list_available_providers(self) -> List[str]:
        """List names of available (enabled and configured) providers."""
        available = []
        for name, config in self._provider_configs.items():
            if config.enabled and config.api_key:
                available.append(name)
        return available
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.MODEL_MAPPINGS.keys())


def load_llm_from_name(
    model_name: str,
    registry: Optional[ProviderRegistry] = None,
    config_loader: Optional[LLMConfigLoader] = None,
) -> Optional[LLMProviderBase]:
    """
    Load an LLM provider by friendly model name.
    
    Convenience function for loading providers using friendly names like
    "gpt-4", "claude-3", "gemini-pro", etc.
    
    Args:
        model_name: Friendly model name
        registry: Optional existing registry
        config_loader: Optional config loader (creates new one if None)
        
    Returns:
        Provider instance or None if not found
    """
    if config_loader is None:
        config_loader = LLMConfigLoader()
    
    mapping = config_loader.MODEL_MAPPINGS.get(model_name.lower())
    if not mapping:
        return None
    
    provider_name = mapping["provider"]
    return config_loader.create_provider(provider_name)


# Global instance for convenience
_default_loader: Optional[LLMConfigLoader] = None


def get_default_loader() -> LLMConfigLoader:
    """Get or create default config loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = LLMConfigLoader()
    return _default_loader

