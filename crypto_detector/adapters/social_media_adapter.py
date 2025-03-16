from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class SocialMediaAdapter(ABC):
    """
    Abstract base class for social media adapters.
    All social media platform implementations must inherit from this class.
    """

    @abstractmethod
    async def get_current_mentions(self, coin_name: str) -> int:
        """
        Get the current number of mentions for a specific cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Number of mentions
        """
        pass

    @abstractmethod
    async def get_historical_mentions(self, coin_name: str) -> Dict[str, int]:
        """
        Get historical mentions data for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary mapping datetime strings (format: "YYYY-MM-DD HH") to mention counts
        """
        pass

    async def get_sentiment(self, coin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis for a cryptocurrency (optional method)

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary with sentiment information or None if not supported
        """
        # Default implementation returns None - subclasses can override if they support sentiment analysis
        return None

    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this social media adapter

        :return: Name of the social media platform
        """
        pass
