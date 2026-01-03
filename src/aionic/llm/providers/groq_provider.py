"""
Groq Provider - Implementation for Groq API.

Supports fast inference with various open-source models.
"""

import os
from typing import List, Optional, Dict, Any

from ..base import LLMProviderBase
from ..models import (
    ModelConfig,
    LLMResponse,
    ChatMessage,
    EmbeddingResponse,
    ProviderMetadata,
    MessageRole,
)


class GroqProvider(LLMProviderBase):
    """
    Groq API provider implementation.
    
    Supports:
    - Chat completion (Llama, Mixtral, etc.)
    - Fast inference with optimized hardware
    
    API Key Configuration:
    - Environment variable: GROQ_API_KEY
    - Constructor parameter: api_key
    - Config dict: {"api_key": "..."}
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Groq provider."""
        super().__init__(api_key, config)
        self._client = None
        self._api_key = api_key
    
    def initialize(self) -> None:
        """Initialize Groq client."""
        # Try to get API key from various sources
        if not self._api_key:
            self._api_key = os.getenv("GROQ_API_KEY")
        
        if not self._api_key:
            # Check config dict
            if self.config and "api_key" in self.config:
                self._api_key = self.config["api_key"]
        
        if not self._api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Lazy import
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "Groq package not installed. Install with: pip install groq"
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate text completion (uses chat interface)."""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        return self.chat(messages, config)
    
    def chat(
        self,
        messages: List[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate chat response."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        config = config or ModelConfig(model="llama3-8b-8192")
        
        try:
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                groq_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
            
            # Prepare parameters
            params = {
                "model": config.model,
                "messages": groq_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            # Call API
            response = self._client.chat.completions.create(**params)
            
            # Extract response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                finish_reason = response.choices[0].finish_reason
            else:
                content = ""
                usage = {}
                finish_reason = None
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="groq",
                usage=usage,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")
    
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings (not supported by Groq)."""
        raise NotImplementedError(
            "Groq API does not support embeddings. Use OpenAI or another provider."
        )
    
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        return ProviderMetadata(
            name="groq",
            display_name="Groq",
            supported_models=[
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it",
            ],
            supports_chat=True,
            supports_completion=True,
            supports_embeddings=False,
            cost_per_1k_tokens=0.0001,  # Very low cost
            cost_per_1k_tokens_output=0.0001,
            avg_latency_ms=100.0,  # Very fast
            max_tokens=8192,
        )

