"""
LLM Router - Intelligent routing and failover for LLM providers.

Implements routing strategies for selecting optimal providers based on
cost, speed, quality, or failover requirements.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

from .base import LLMProviderBase
from .registry import ProviderRegistry
from .models import ModelConfig, LLMResponse, ChatMessage, EmbeddingResponse
from ..memory.audit_logger import AuditLogger, LogLevel, LogCategory


class RoutingStrategy(Enum):
    """Routing strategies for provider selection."""
    COST_OPTIMIZED = "cost_optimized"      # Minimize cost
    SPEED_OPTIMIZED = "speed_optimized"    # Minimize latency
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
    PROVIDER_FALLBACK = "provider_fallback"   # Try primary, fallback on failure


class LLMRouter:
    """
    Intelligent router for LLM provider selection and failover.
    
    Features:
    - Strategy-based provider selection
    - Automatic failover on errors
    - Cost and latency tracking
    - Integration with audit logging
    - Task metadata-aware routing
    
    Design:
    - Agents use router, never providers directly
    - Router abstracts provider selection logic
    - Failover is transparent to agents
    - All routing decisions are logged
    """
    
    def __init__(
        self,
        registry: ProviderRegistry,
        audit_logger: Optional[AuditLogger] = None,
        default_strategy: RoutingStrategy = RoutingStrategy.QUALITY_OPTIMIZED,
    ):
        """
        Initialize router.
        
        Args:
            registry: Provider registry
            audit_logger: Optional audit logger for routing decisions
            default_strategy: Default routing strategy
        """
        self.registry = registry
        self.audit_logger = audit_logger
        self.default_strategy = default_strategy
        
        # Performance tracking
        self._provider_stats: Dict[str, Dict[str, Any]] = {}
        
        # Default provider preferences (can be overridden)
        self._provider_preferences: Dict[RoutingStrategy, List[str]] = {
            RoutingStrategy.COST_OPTIMIZED: [],
            RoutingStrategy.SPEED_OPTIMIZED: [],
            RoutingStrategy.QUALITY_OPTIMIZED: [],
            RoutingStrategy.PROVIDER_FALLBACK: [],
        }
    
    def set_provider_preferences(
        self,
        strategy: RoutingStrategy,
        provider_names: List[str],
    ) -> None:
        """
        Set provider preference order for a strategy.
        
        Args:
            strategy: Routing strategy
            provider_names: Ordered list of provider names (first is preferred)
        """
        self._provider_preferences[strategy] = provider_names
    
    def _select_provider(
        self,
        strategy: RoutingStrategy,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LLMProviderBase]:
        """
        Select provider based on strategy.
        
        Args:
            strategy: Routing strategy to use
            task_metadata: Optional task metadata for context
            
        Returns:
            Selected provider or None if none available
        """
        available_providers = self.registry.get_available_providers()
        if not available_providers:
            return None
        
        # Check for explicit preferences
        preferences = self._provider_preferences.get(strategy, [])
        if preferences:
            for name in preferences:
                if name in available_providers:
                    provider = self.registry.get(name)
                    if provider and provider.is_available():
                        return provider
        
        # Strategy-based selection
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._select_cost_optimized(available_providers)
        elif strategy == RoutingStrategy.SPEED_OPTIMIZED:
            return self._select_speed_optimized(available_providers)
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return self._select_quality_optimized(available_providers)
        elif strategy == RoutingStrategy.PROVIDER_FALLBACK:
            # Use first available provider as primary
            if available_providers:
                return self.registry.get(available_providers[0])
        
        # Default: first available
        if available_providers:
            return self.registry.get(available_providers[0])
        
        return None
    
    def _select_cost_optimized(self, available: List[str]) -> Optional[LLMProviderBase]:
        """Select provider with lowest cost."""
        best_provider = None
        best_cost = float('inf')
        
        for name in available:
            provider = self.registry.get(name)
            if not provider:
                continue
            
            metadata = provider.metadata()
            # Use input cost as primary metric
            cost = metadata.cost_per_1k_tokens
            
            if cost < best_cost:
                best_cost = cost
                best_provider = provider
        
        return best_provider
    
    def _select_speed_optimized(self, available: List[str]) -> Optional[LLMProviderBase]:
        """Select provider with lowest latency."""
        best_provider = None
        best_latency = float('inf')
        
        for name in available:
            provider = self.registry.get(name)
            if not provider:
                continue
            
            metadata = provider.metadata()
            latency = metadata.avg_latency_ms
            
            # Consider historical performance if available
            stats = self._provider_stats.get(name, {})
            avg_latency = stats.get('avg_latency_ms', latency)
            
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_provider = provider
        
        return best_provider
    
    def _select_quality_optimized(self, available: List[str]) -> Optional[LLMProviderBase]:
        """Select provider with best quality (typically most capable models)."""
        # For quality, prefer providers with larger context windows and newer models
        best_provider = None
        best_score = -1
        
        for name in available:
            provider = self.registry.get(name)
            if not provider:
                continue
            
            metadata = provider.metadata()
            # Score based on max tokens and model capabilities
            score = metadata.max_tokens / 1000.0  # Normalize
            
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    def _log_routing_decision(
        self,
        provider_name: str,
        strategy: RoutingStrategy,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log routing decision to audit logger."""
        if self.audit_logger:
            level = LogLevel.INFO if success else LogLevel.WARNING
            message = f"LLM routing: {strategy.value} -> {provider_name}"
            if error:
                message += f" (error: {error})"
            
            self.audit_logger.log(
                level=level,
                category=LogCategory.SYSTEM,
                message=message,
                metadata={
                    "provider": provider_name,
                    "strategy": strategy.value,
                    "success": success,
                    "error": error,
                },
            )
    
    def _update_provider_stats(
        self,
        provider_name: str,
        latency_ms: float,
        tokens: int,
        cost: float,
    ) -> None:
        """Update provider performance statistics."""
        if provider_name not in self._provider_stats:
            self._provider_stats[provider_name] = {
                "total_requests": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
        
        stats = self._provider_stats[provider_name]
        stats["total_requests"] += 1
        stats["total_latency_ms"] += latency_ms
        stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
        stats["total_tokens"] += tokens
        stats["total_cost"] += cost
    
    def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None,
        strategy: Optional[RoutingStrategy] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> LLMResponse:
        """
        Generate text with automatic provider selection and failover.
        
        Args:
            prompt: Input prompt
            config: Model configuration
            strategy: Routing strategy (uses default if None)
            task_metadata: Optional task metadata for routing
            max_retries: Maximum retry attempts with different providers
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            RuntimeError: If all providers fail
        """
        strategy = strategy or self.default_strategy
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            provider = self._select_provider(strategy, task_metadata)
            if not provider:
                raise RuntimeError("No available LLM providers")
            
            provider_name = provider.metadata().name
            start_time = datetime.utcnow()
            
            try:
                response = provider.generate(prompt, config)
                
                # Calculate metrics
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                tokens = response.usage.get("total_tokens", 0)
                cost = provider.estimate_cost(
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                )
                
                self._update_provider_stats(provider_name, latency_ms, tokens, cost)
                self._log_routing_decision(provider_name, strategy, True)
                
                return response
                
            except Exception as e:
                last_error = str(e)
                self._log_routing_decision(provider_name, strategy, False, last_error)
                
                # Try next provider if available
                retries += 1
                if retries <= max_retries:
                    # Switch to fallback strategy for retries
                    strategy = RoutingStrategy.PROVIDER_FALLBACK
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def chat(
        self,
        messages: List[ChatMessage],
        config: Optional[ModelConfig] = None,
        strategy: Optional[RoutingStrategy] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> LLMResponse:
        """
        Chat with automatic provider selection and failover.
        
        Args:
            messages: Chat message history
            config: Model configuration
            strategy: Routing strategy (uses default if None)
            task_metadata: Optional task metadata for routing
            max_retries: Maximum retry attempts with different providers
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            RuntimeError: If all providers fail
        """
        strategy = strategy or self.default_strategy
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            provider = self._select_provider(strategy, task_metadata)
            if not provider:
                raise RuntimeError("No available LLM providers")
            
            provider_name = provider.metadata().name
            start_time = datetime.utcnow()
            
            try:
                response = provider.chat(messages, config)
                
                # Calculate metrics
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                tokens = response.usage.get("total_tokens", 0)
                cost = provider.estimate_cost(
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                )
                
                self._update_provider_stats(provider_name, latency_ms, tokens, cost)
                self._log_routing_decision(provider_name, strategy, True)
                
                return response
                
            except Exception as e:
                last_error = str(e)
                self._log_routing_decision(provider_name, strategy, False, last_error)
                
                retries += 1
                if retries <= max_retries:
                    strategy = RoutingStrategy.PROVIDER_FALLBACK
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def embed(
        self,
        text: str,
        config: Optional[ModelConfig] = None,
        strategy: Optional[RoutingStrategy] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> EmbeddingResponse:
        """
        Generate embeddings with automatic provider selection and failover.
        
        Args:
            text: Input text to embed
            config: Model configuration
            strategy: Routing strategy (uses default if None)
            task_metadata: Optional task metadata for routing
            max_retries: Maximum retry attempts with different providers
            
        Returns:
            EmbeddingResponse with embedding vectors
            
        Raises:
            RuntimeError: If all providers fail or none support embeddings
        """
        strategy = strategy or self.default_strategy
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            provider = self._select_provider(strategy, task_metadata)
            if not provider:
                raise RuntimeError("No available LLM providers")
            
            # Check if provider supports embeddings
            metadata = provider.metadata()
            if not metadata.supports_embeddings:
                retries += 1
                if retries <= max_retries:
                    continue
                raise RuntimeError("No providers support embeddings")
            
            provider_name = metadata.name
            start_time = datetime.utcnow()
            
            try:
                response = provider.embeddings(text, config)
                
                # Calculate metrics
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                tokens = response.usage.get("total_tokens", 0)
                cost = provider.estimate_cost(tokens, 0)
                
                self._update_provider_stats(provider_name, latency_ms, tokens, cost)
                self._log_routing_decision(provider_name, strategy, True)
                
                return response
                
            except Exception as e:
                last_error = str(e)
                self._log_routing_decision(provider_name, strategy, False, last_error)
                
                retries += 1
                if retries <= max_retries:
                    strategy = RoutingStrategy.PROVIDER_FALLBACK
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        return {
            "provider_stats": self._provider_stats.copy(),
            "default_strategy": self.default_strategy.value,
            "available_providers": self.registry.get_available_providers(),
        }

