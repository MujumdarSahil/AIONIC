"""
OpenAI Provider - Implementation for OpenAI API.

Supports GPT-3.5, GPT-4, and other OpenAI models.
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


class OpenAIProvider(LLMProviderBase):
    """
    OpenAI API provider implementation.
    
    Supports:
    - Chat completion (GPT-3.5, GPT-4)
    - Text completion (legacy)
    - Embeddings (text-embedding-ada-002, etc.)
    
    API Key Configuration:
    - Environment variable: OPENAI_API_KEY
    - Constructor parameter: api_key
    - Config dict: {"api_key": "..."}
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI provider."""
        super().__init__(api_key, config)
        self._client = None
        self._api_key = api_key
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        # Try to get API key from various sources
        if not self._api_key:
            self._api_key = os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            # Check config dict
            if self.config and "api_key" in self.config:
                self._api_key = self.config["api_key"]
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Lazy import to avoid requiring openai package if not used
        try:
            import openai
            openai.api_key = self._api_key
            
            # Try to create client (works with both old and new SDK versions)
            try:
                self._client = openai.OpenAI(api_key=self._api_key)
            except (AttributeError, TypeError):
                # Fallback for older SDK versions
                self._client = openai
            
            self._initialized = True
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate text completion."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        config = config or ModelConfig(model="gpt-3.5-turbo")
        
        try:
            import openai
            
            # Use chat completion for better compatibility
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare parameters
            params = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            
            if config.top_p != 1.0:
                params["top_p"] = config.top_p
            if config.frequency_penalty != 0.0:
                params["frequency_penalty"] = config.frequency_penalty
            if config.presence_penalty != 0.0:
                params["presence_penalty"] = config.presence_penalty
            if config.stop:
                params["stop"] = config.stop
            
            # Call API
            if hasattr(self._client, 'chat'):
                response = self._client.chat.completions.create(**params)
            else:
                # Fallback for older SDK
                response = self._client.ChatCompletion.create(**params)
            
            # Extract response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
            else:
                content = str(response)
                usage = {}
                finish_reason = None
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="openai",
                usage=usage,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def chat(
        self,
        messages: List[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate chat response."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        config = config or ModelConfig(model="gpt-3.5-turbo")
        
        try:
            import openai
            
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_msg = {"role": msg.role.value, "content": msg.content}
                if msg.name:
                    openai_msg["name"] = msg.name
                if msg.function_call:
                    openai_msg["function_call"] = msg.function_call
                openai_messages.append(openai_msg)
            
            # Prepare parameters
            params = {
                "model": config.model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            
            if config.top_p != 1.0:
                params["top_p"] = config.top_p
            if config.frequency_penalty != 0.0:
                params["frequency_penalty"] = config.frequency_penalty
            if config.presence_penalty != 0.0:
                params["presence_penalty"] = config.presence_penalty
            if config.stop:
                params["stop"] = config.stop
            
            # Call API
            if hasattr(self._client, 'chat'):
                response = self._client.chat.completions.create(**params)
            else:
                response = self._client.ChatCompletion.create(**params)
            
            # Extract response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
            else:
                content = str(response)
                usage = {}
                finish_reason = None
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="openai",
                usage=usage,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        config = config or ModelConfig(model="text-embedding-ada-002")
        
        try:
            import openai
            
            # Call embeddings API
            if hasattr(self._client, 'embeddings'):
                response = self._client.embeddings.create(
                    model=config.model,
                    input=text,
                )
            else:
                response = self._client.Embedding.create(
                    model=config.model,
                    input=text,
                )
            
            # Extract embeddings
            if hasattr(response, 'data') and len(response.data) > 0:
                embeddings = [item.embedding for item in response.data]
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
            else:
                embeddings = []
                usage = {}
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=config.model,
                provider="openai",
                usage=usage,
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings API error: {e}")
    
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        return ProviderMetadata(
            name="openai",
            display_name="OpenAI",
            supported_models=[
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
            supports_chat=True,
            supports_completion=True,
            supports_embeddings=True,
            cost_per_1k_tokens=0.002,  # Approximate for gpt-3.5-turbo
            cost_per_1k_tokens_output=0.002,
            avg_latency_ms=500.0,
            max_tokens=4096,
        )

