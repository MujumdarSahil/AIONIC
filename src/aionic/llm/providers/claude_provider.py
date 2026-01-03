"""
Claude Provider - Implementation for Anthropic Claude API.

Supports Claude 3 Opus, Sonnet, and Haiku models.
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


class ClaudeProvider(LLMProviderBase):
    """
    Anthropic Claude API provider implementation.
    
    Supports:
    - Chat completion (Claude 3 Opus, Sonnet, Haiku)
    
    Note: Claude API does not support embeddings or text completion.
    
    API Key Configuration:
    - Environment variable: ANTHROPIC_API_KEY
    - Constructor parameter: api_key
    - Config dict: {"api_key": "..."}
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Claude provider."""
        super().__init__(api_key, config)
        self._client = None
        self._api_key = api_key
    
    def initialize(self) -> None:
        """Initialize Anthropic client."""
        # Try to get API key from various sources
        if not self._api_key:
            self._api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self._api_key:
            # Check config dict
            if self.config and "api_key" in self.config:
                self._api_key = self.config["api_key"]
        
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Lazy import
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate text completion (uses chat interface)."""
        # Claude only supports chat, so convert prompt to user message
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
        
        config = config or ModelConfig(model="claude-3-sonnet-20240229")
        
        try:
            # Convert messages to Anthropic format
            # Anthropic uses system/user/assistant format
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                elif msg.role == MessageRole.USER:
                    anthropic_messages.append({
                        "role": "user",
                        "content": msg.content,
                    })
                elif msg.role == MessageRole.ASSISTANT:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": msg.content,
                    })
            
            # Prepare parameters
            params = {
                "model": config.model,
                "messages": anthropic_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 1024,
            }
            
            if system_message:
                params["system"] = system_message
            
            # Call API
            response = self._client.messages.create(**params)
            
            # Extract response
            if response.content and len(response.content) > 0:
                content = response.content[0].text
            else:
                content = ""
            
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            
            finish_reason = response.stop_reason if hasattr(response, 'stop_reason') else None
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="claude",
                usage=usage,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings (not supported by Claude)."""
        raise NotImplementedError(
            "Claude API does not support embeddings. Use OpenAI or another provider."
        )
    
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        return ProviderMetadata(
            name="claude",
            display_name="Anthropic Claude",
            supported_models=[
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
            ],
            supports_chat=True,
            supports_completion=False,  # Only chat interface
            supports_embeddings=False,
            cost_per_1k_tokens=0.003,  # Approximate for claude-3-sonnet
            cost_per_1k_tokens_output=0.015,
            avg_latency_ms=800.0,
            max_tokens=200000,  # Claude 3 has large context window
        )

