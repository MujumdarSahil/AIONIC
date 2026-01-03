"""
Provider Registry - Central registry for LLM providers.

Manages provider registration, discovery, and lifecycle.
"""

from typing import Dict, List, Optional, Type
from .base import LLMProviderBase
from .models import ProviderMetadata


class ProviderRegistry:
    """
    Central registry for LLM providers.
    
    Manages provider instances and provides discovery capabilities.
    Supports runtime registration and provider lookup.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._providers: Dict[str, LLMProviderBase] = {}
        self._provider_classes: Dict[str, Type[LLMProviderBase]] = {}
    
    def register(
        self,
        provider: LLMProviderBase,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a provider instance.
        
        Args:
            provider: Provider instance to register
            name: Optional custom name (uses provider.metadata().name if None)
            
        Raises:
            ValueError: If provider is invalid or name conflicts
        """
        if not isinstance(provider, LLMProviderBase):
            raise ValueError("Provider must be an instance of LLMProviderBase")
        
        # Get provider name
        if name is None:
            metadata = provider.metadata()
            name = metadata.name
        
        if name in self._providers:
            raise ValueError(f"Provider '{name}' is already registered")
        
        # Initialize provider if not already initialized
        if not provider.is_available():
            try:
                provider.initialize()
            except Exception as e:
                raise ValueError(f"Failed to initialize provider '{name}': {e}")
        
        self._providers[name] = provider
    
    def register_class(
        self,
        provider_class: Type[LLMProviderBase],
        name: Optional[str] = None,
    ) -> None:
        """
        Register a provider class for lazy instantiation.
        
        Args:
            provider_class: Provider class to register
            name: Optional custom name
        """
        if not issubclass(provider_class, LLMProviderBase):
            raise ValueError("Provider class must be a subclass of LLMProviderBase")
        
        if name is None:
            # Try to get name from a temporary instance
            try:
                temp_instance = provider_class()
                metadata = temp_instance.metadata()
                name = metadata.name
            except Exception:
                name = provider_class.__name__.lower().replace("provider", "")
        
        self._provider_classes[name] = provider_class
    
    def get(self, name: str) -> Optional[LLMProviderBase]:
        """
        Get provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(name)
    
    def create_provider(
        self,
        name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> Optional[LLMProviderBase]:
        """
        Create provider instance from registered class.
        
        Args:
            name: Provider name
            api_key: Optional API key
            config: Optional configuration
            
        Returns:
            Provider instance or None if class not found
        """
        provider_class = self._provider_classes.get(name)
        if provider_class is None:
            return None
        
        provider = provider_class(api_key=api_key, config=config)
        provider.initialize()
        return provider
    
    def list_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def list_provider_classes(self) -> List[str]:
        """
        List all registered provider class names.
        
        Returns:
            List of provider class names
        """
        return list(self._provider_classes.keys())
    
    def get_metadata(self, name: str) -> Optional[ProviderMetadata]:
        """
        Get metadata for a provider.
        
        Args:
            name: Provider name
            
        Returns:
            ProviderMetadata or None if not found
        """
        provider = self.get(name)
        if provider is None:
            return None
        return provider.metadata()
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a provider.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider was removed, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._provider_classes.clear()
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available (initialized) providers.
        
        Returns:
            List of provider names that are available
        """
        return [
            name for name, provider in self._providers.items()
            if provider.is_available()
        ]

