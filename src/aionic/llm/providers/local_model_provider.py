"""
Local Model Provider - Stub for local/self-hosted models.

This is a placeholder implementation for integrating local models
(e.g., via Ollama, vLLM, or other local inference servers).
"""

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


class LocalModelProvider(LLMProviderBase):
    """
    Local model provider stub implementation.
    
    This is a placeholder for integrating local models. Users should
    extend this class to implement their specific local model integration
    (e.g., Ollama, vLLM, Transformers, etc.).
    
    Example integration points:
    - Ollama: Use ollama Python client
    - vLLM: Use vLLM API server
    - Transformers: Direct model loading
    - Custom API: HTTP client to local server
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize local model provider."""
        super().__init__(api_key, config)
        self._base_url = config.get("base_url", "http://localhost:11434") if config else "http://localhost:11434"
        self._model_name = config.get("model_name", "llama2") if config else "llama2"
    
    def initialize(self) -> None:
        """
        Initialize local model connection.
        
        This is a stub. Override this method to implement actual
        connection logic for your local model setup.
        """
        # Stub implementation - should be overridden
        self._initialized = True
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """
        Generate text completion.
        
        This is a stub. Override this method to implement actual
        generation logic for your local model.
        """
        raise NotImplementedError(
            "LocalModelProvider is a stub. Extend this class and implement "
            "generate() for your local model setup (e.g., Ollama, vLLM)."
        )
    
    def chat(
        self,
        messages: List[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """
        Generate chat response.
        
        This is a stub. Override this method to implement actual
        chat logic for your local model.
        """
        raise NotImplementedError(
            "LocalModelProvider is a stub. Extend this class and implement "
            "chat() for your local model setup."
        )
    
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings.
        
        This is a stub. Override this method to implement actual
        embedding logic for your local model.
        """
        raise NotImplementedError(
            "LocalModelProvider is a stub. Extend this class and implement "
            "embeddings() for your local model setup."
        )
    
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        return ProviderMetadata(
            name="local",
            display_name="Local Model",
            supported_models=[self._model_name],
            supports_chat=True,
            supports_completion=True,
            supports_embeddings=False,
            cost_per_1k_tokens=0.0,  # No API cost for local
            cost_per_1k_tokens_output=0.0,
            avg_latency_ms=200.0,  # Depends on hardware
            max_tokens=4096,
        )

