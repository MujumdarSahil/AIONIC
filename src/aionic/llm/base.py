"""
LLM Provider Base - Abstract interface for LLM providers.

Defines the contract that all LLM provider implementations must follow,
ensuring vendor-agnostic agent logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .models import (
    ModelConfig,
    LLMResponse,
    ChatMessage,
    EmbeddingResponse,
    ProviderMetadata,
)


class LLMProviderBase(ABC):
    """
    Abstract base class for all LLM providers.
    
    All provider implementations must inherit from this class and
    implement the required methods. This ensures agents can work
    with any LLM provider without vendor-specific code.
    
    Design Principles:
    - Agents never depend on specific providers
    - Configuration is runtime-pluggable
    - Error handling is consistent across providers
    - Metadata is standardized
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize provider.
        
        Args:
            api_key: API key for the provider (can be None if using env vars)
            config: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize provider connection and validate credentials.
        
        This method should:
        - Load API keys from environment if not provided
        - Validate credentials
        - Set up any required client connections
        - Raise exceptions on failure
        
        Raises:
            ValueError: If API key is missing or invalid
            ConnectionError: If provider connection fails
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: Input prompt text
            config: Model configuration (uses defaults if None)
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            ValueError: If prompt is invalid
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of chat messages (conversation history)
            config: Model configuration (uses defaults if None)
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            ValueError: If messages are invalid
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text to embed
            config: Model configuration (uses defaults if None)
            
        Returns:
            EmbeddingResponse with embedding vectors
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding generation fails
            NotImplementedError: If provider doesn't support embeddings
        """
        pass
    
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """
        Get provider metadata and capabilities.
        
        Returns:
            ProviderMetadata with provider information
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if provider is available and initialized.
        
        Returns:
            True if provider is ready to use
        """
        return self._initialized
    
    def validate_config(self, config: ModelConfig) -> bool:
        """
        Validate model configuration for this provider.
        
        Args:
            config: Model configuration to validate
            
        Returns:
            True if configuration is valid
        """
        metadata = self.metadata()
        return config.model in metadata.supported_models
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        metadata = self.metadata()
        input_cost = (input_tokens / 1000.0) * metadata.cost_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * metadata.cost_per_1k_tokens_output
        return input_cost + output_cost

