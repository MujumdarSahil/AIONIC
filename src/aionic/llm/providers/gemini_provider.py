"""
Gemini Provider - Implementation for Google Gemini API.

Supports Gemini Pro, Gemini Ultra, and other Google models.
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


class GeminiProvider(LLMProviderBase):
    """
    Google Gemini API provider implementation.
    
    Supports:
    - Chat completion (Gemini Pro, Ultra)
    - Embeddings (text-embedding-004)
    
    API Key Configuration:
    - Environment variable: GOOGLE_API_KEY
    - Constructor parameter: api_key
    - Config dict: {"api_key": "..."}
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Gemini provider."""
        super().__init__(api_key, config)
        self._client = None
        self._api_key = api_key
    
    def initialize(self) -> None:
        """Initialize Google Gemini client."""
        # Try to get API key from various sources
        if not self._api_key:
            self._api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self._api_key:
            # Check config dict
            if self.config and "api_key" in self.config:
                self._api_key = self.config["api_key"]
        
        if not self._api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Lazy import
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai
            self._initialized = True
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
    ) -> LLMResponse:
        """Generate text completion."""
        # Convert to chat format
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
        
        config = config or ModelConfig(model="gemini-pro")
        
        try:
            import google.generativeai as genai
            
            # Get model
            model = genai.GenerativeModel(config.model)
            
            # Convert messages to Gemini format
            # Gemini uses a simple list of message parts
            chat_history = []
            for msg in messages:
                if msg.role == MessageRole.USER:
                    chat_history.append({"role": "user", "parts": [msg.content]})
                elif msg.role == MessageRole.ASSISTANT:
                    chat_history.append({"role": "model", "parts": [msg.content]})
            
            # Start chat if we have history
            if len(chat_history) > 1:
                chat = model.start_chat(history=chat_history[:-1])
                user_message = chat_history[-1]["parts"][0]
            else:
                chat = model.start_chat()
                user_message = messages[-1].content if messages else ""
            
            # Generate response
            generation_config = {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            response = chat.send_message(
                user_message,
                generation_config=generation_config,
            )
            
            # Extract response
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Estimate token usage (Gemini doesn't always provide this)
            usage = {
                "prompt_tokens": len(user_message.split()) * 1.3,  # Rough estimate
                "completion_tokens": len(content.split()) * 1.3,
                "total_tokens": 0,
            }
            usage["total_tokens"] = int(usage["prompt_tokens"] + usage["completion_tokens"])
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="gemini",
                usage=usage,
                finish_reason=None,
            )
            
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
    
    def embeddings(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
        
        config = config or ModelConfig(model="text-embedding-004")
        
        try:
            import google.generativeai as genai
            
            # Use embedding model
            result = genai.embed_content(
                model=config.model,
                content=text,
            )
            
            # Extract embeddings
            if hasattr(result, 'embedding'):
                embeddings = [result.embedding]
            elif isinstance(result, dict) and 'embedding' in result:
                embeddings = [result['embedding']]
            else:
                embeddings = []
            
            usage = {
                "prompt_tokens": len(text.split()) * 1.3,  # Estimate
                "total_tokens": 0,
            }
            usage["total_tokens"] = int(usage["prompt_tokens"])
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=config.model,
                provider="gemini",
                usage=usage,
            )
            
        except Exception as e:
            raise RuntimeError(f"Gemini embeddings API error: {e}")
    
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata."""
        return ProviderMetadata(
            name="gemini",
            display_name="Google Gemini",
            supported_models=[
                "gemini-pro",
                "gemini-pro-vision",
                "gemini-ultra",
                "text-embedding-004",
            ],
            supports_chat=True,
            supports_completion=True,
            supports_embeddings=True,
            cost_per_1k_tokens=0.0005,  # Approximate
            cost_per_1k_tokens_output=0.0015,
            avg_latency_ms=600.0,
            max_tokens=32768,
        )

