import os
import logging
from typing import Optional, Dict, List, Type

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter
from crypto_detector.adapters.twitter_social_adapter import TwitterAdapter
from crypto_detector.adapters.reddit_social_adapter import RedditAdapter
from crypto_detector.adapters.telegram_social_adapter import TelegramAdapter

# Setup logging
logger = logging.getLogger(__name__)


class SocialMediaFactory:
    """
    Factory class for creating social media adapters.
    Provides a centralized way to register, create, and manage adapters.
    """

    # Mapping of platform names to adapter classes
    _adapter_classes = {
        'twitter': TwitterAdapter,
        'reddit': RedditAdapter,
        'telegram': TelegramAdapter
    }

    # Registry of custom adapter classes
    _custom_adapters = {}

    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[SocialMediaAdapter]):
        """
        Register a custom social media adapter

        :param name: Name of the platform
        :param adapter_class: Adapter class that implements SocialMediaAdapter
        """
        if not issubclass(adapter_class, SocialMediaAdapter):
            raise TypeError("Adapter class must inherit from SocialMediaAdapter")

        cls._custom_adapters[name] = adapter_class
        logger.info(f"Custom adapter '{name}' registered")

    @classmethod
    def create_adapter(cls, platform: str, **kwargs) -> Optional[SocialMediaAdapter]:
        """
        Create an adapter for a specific platform

        :param platform: Platform name
        :param kwargs: Additional arguments to pass to the adapter constructor
        :return: Adapter instance or None if platform not supported
        """
        # Check for custom adapters first
        if platform in cls._custom_adapters:
            try:
                adapter = cls._custom_adapters[platform](**kwargs)
                logger.info(f"Created custom {platform} adapter")
                return adapter
            except Exception as e:
                logger.error(f"Failed to create custom {platform} adapter: {str(e)}")
                return None

        # Check for built-in adapters
        if platform in cls._adapter_classes:
            try:
                adapter = cls._adapter_classes[platform](**kwargs)
                logger.info(f"Created {platform} adapter")
                return adapter
            except Exception as e:
                logger.error(f"Failed to create {platform} adapter: {str(e)}")
                return None

        logger.warning(f"Unsupported platform: {platform}")
        return None

    @classmethod
    def create_all_adapters(cls, config: Dict = None) -> Dict[str, SocialMediaAdapter]:
        """
        Create all available adapters

        :param config: Configuration for adapters
        :return: Dictionary mapping platform names to adapter instances
        """
        adapters = {}
        config = config or {}

        # Create built-in adapters
        for platform, adapter_class in cls._adapter_classes.items():
            platform_config = config.get(platform, {})
            try:
                adapter = adapter_class(**platform_config)
                adapters[platform] = adapter
                logger.info(f"Created {platform} adapter")
            except Exception as e:
                logger.error(f"Failed to create {platform} adapter: {str(e)}")

        # Create custom adapters
        for platform, adapter_class in cls._custom_adapters.items():
            # Skip if we already created this platform
            if platform in adapters:
                continue

            platform_config = config.get(platform, {})
            try:
                adapter = adapter_class(**platform_config)
                adapters[platform] = adapter
                logger.info(f"Created custom {platform} adapter")
            except Exception as e:
                logger.error(f"Failed to create custom {platform} adapter: {str(e)}")

        return adapters

    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """
        Get list of all available platforms

        :return: List of platform names
        """
        return list(set(list(cls._adapter_classes.keys()) + list(cls._custom_adapters.keys())))
