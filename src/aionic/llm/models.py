"""
LLM Models - Data structures for LLM configuration and responses.

Defines schemas for model configuration, responses, and provider metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MessageRole(Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """
    Single chat message in a conversation.
    
    Attributes:
        role: Message role (system, user, assistant, function)
        content: Message content
        name: Optional name for function calls
        function_call: Optional function call data
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        return result


@dataclass
class ModelConfig:
    """
    Configuration for LLM model usage.
    
    Attributes:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        stop: Stop sequences
        stream: Whether to stream responses
        metadata: Additional provider-specific metadata
    """
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
            **self.metadata,
        }
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop is not None:
            result["stop"] = self.stop
        return result


@dataclass
class LLMResponse:
    """
    Response from LLM generation or chat.
    
    Attributes:
        content: Generated text content
        model: Model used for generation
        provider: Provider name (e.g., "openai", "claude")
        usage: Token usage information
        finish_reason: Reason for completion (stop, length, etc.)
        metadata: Additional response metadata
    """
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


@dataclass
class EmbeddingResponse:
    """
    Response from embedding generation.
    
    Attributes:
        embeddings: List of embedding vectors
        model: Model used for embedding
        provider: Provider name
        usage: Token usage information
        metadata: Additional response metadata
    """
    embeddings: List[List[float]]
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "embeddings": self.embeddings,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "metadata": self.metadata,
        }


@dataclass
class ProviderMetadata:
    """
    Metadata about an LLM provider.
    
    Attributes:
        name: Provider name (e.g., "openai", "claude")
        display_name: Human-readable name
        supported_models: List of supported model identifiers
        supports_chat: Whether provider supports chat interface
        supports_completion: Whether provider supports completion interface
        supports_embeddings: Whether provider supports embeddings
        cost_per_1k_tokens: Estimated cost per 1k tokens (input)
        cost_per_1k_tokens_output: Estimated cost per 1k tokens (output)
        avg_latency_ms: Average latency in milliseconds
        max_tokens: Maximum tokens supported
    """
    name: str
    display_name: str
    supported_models: List[str]
    supports_chat: bool = True
    supports_completion: bool = True
    supports_embeddings: bool = False
    cost_per_1k_tokens: float = 0.0
    cost_per_1k_tokens_output: float = 0.0
    avg_latency_ms: float = 0.0
    max_tokens: int = 4096
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "supported_models": self.supported_models,
            "supports_chat": self.supports_chat,
            "supports_completion": self.supports_completion,
            "supports_embeddings": self.supports_embeddings,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "cost_per_1k_tokens_output": self.cost_per_1k_tokens_output,
            "avg_latency_ms": self.avg_latency_ms,
            "max_tokens": self.max_tokens,
        }

