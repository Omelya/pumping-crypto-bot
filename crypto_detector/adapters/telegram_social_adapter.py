import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter

# Setup logging
logger = logging.getLogger(__name__)


class TelegramAdapter(SocialMediaAdapter):
    """
    Adapter for estimating cryptocurrency mentions on Telegram.

    Note: This is a simplified version that provides estimates instead of actual data.
    A full implementation would require a Telegram client with access to public groups.
    """

    def __init__(self, api_id: Optional[str] = None, api_hash: Optional[str] = None,
                 cache_duration: int = 300):
        """
        Initialize the Telegram adapter

        :param api_id: Telegram API ID (optional, will use env var TELEGRAM_API_ID if not provided)
        :param api_hash: Telegram API hash (optional, will use env var TELEGRAM_API_HASH if not provided)
        :param cache_duration: How long to cache results in seconds (default: 5 minutes)
        """
        self.api_id = api_id or os.getenv('TELEGRAM_API_ID', '')
        self.api_hash = api_hash or os.getenv('TELEGRAM_API_HASH', '')
        self.cache_duration = cache_duration

        # Initialize cache
        self.cache = {}
        self.cache_expiry = {}

        # Crypto group sizes for estimation
        self.group_sizes = {
            'high': [50000, 100000, 200000, 500000, 1000000],  # High popularity coins
            'medium': [10000, 30000, 50000, 70000, 100000],  # Medium popularity coins
            'low': [1000, 5000, 10000, 20000, 30000]  # Low popularity coins
        }

        # Mention rates per 1000 members
        self.mention_rates = {
            'high': [5.0, 10.0, 15.0, 20.0, 25.0],  # High activity periods
            'medium': [1.0, 2.0, 3.0, 5.0, 7.0],  # Medium activity periods
            'low': [0.1, 0.5, 1.0, 1.5, 2.0]  # Low activity periods
        }

        # Coin categories
        self.coin_categories = self._categorize_coins()

        if not self.api_id or not self.api_hash:
            logger.warning("No Telegram API credentials provided. Using estimation mode only.")

    def name(self) -> str:
        """Get the name of this social media adapter"""
        return "Telegram"

    async def get_current_mentions(self, coin_name: str) -> int:
        """
        Get the estimated current number of mentions for a specific cryptocurrency on Telegram

        :param coin_name: Name of the cryptocurrency
        :return: Estimated number of mentions
        """
        # Check cache first
        cache_key = f"current_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        # Get coin popularity category
        popularity = self._get_coin_popularity(coin_name)

        # Get current hour to determine activity level
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 20:  # 8 AM - 8 PM: high activity
            activity = 'high'
        elif 6 <= current_hour <= 22:  # 6 AM - 10 PM: medium activity
            activity = 'medium'
        else:  # Night hours: low activity
            activity = 'low'

        # Estimate number of groups discussing this coin
        num_groups = self._estimate_group_count(coin_name, popularity)

        # Estimate total mentions across all groups
        total_mentions = 0
        for _ in range(num_groups):
            # Choose random group size based on coin popularity
            group_size = np.random.choice(self.group_sizes[popularity])

            # Choose random mention rate based on activity level
            mention_rate = np.random.choice(self.mention_rates[activity])

            # Calculate mentions for this group
            group_mentions = int((group_size / 1000) * mention_rate)
            total_mentions += group_mentions

        # Add some randomness
        if np.random.random() < 0.15:  # 15% chance of a mention spike
            total_mentions *= np.random.uniform(1.5, 3.0)

        # Cache the result
        self.cache[cache_key] = int(total_mentions)
        self.cache_expiry[cache_key] = time.time() + self.cache_duration

        return int(total_mentions)

    async def get_historical_mentions(self, coin_name: str) -> Dict[str, int]:
        """
        Get estimated historical Telegram mentions for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary mapping datetime strings to mention counts
        """
        # Check cache first
        cache_key = f"historical_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        # Get coin popularity category
        popularity = self._get_coin_popularity(coin_name)

        # Create result dictionary with hourly data for the past 7 days
        result = {}
        end_time = datetime.now()

        # Generate data for each hour
        for hour_offset in range(24 * 7, 0, -1):
            hour_time = end_time - timedelta(hours=hour_offset)
            hour_key = hour_time.strftime("%Y-%m-%d %H")

            # Determine activity level based on hour of day
            hour_of_day = hour_time.hour
            if 8 <= hour_of_day <= 20:  # 8 AM - 8 PM: high activity
                activity = 'high'
            elif 6 <= hour_of_day <= 22:  # 6 AM - 10 PM: medium activity
                activity = 'medium'
            else:  # Night hours: low activity
                activity = 'low'

            # Estimate number of groups discussing this coin
            num_groups = self._estimate_group_count(coin_name, popularity)

            # Estimate total mentions across all groups for this hour
            hour_mentions = 0
            for _ in range(num_groups):
                # Choose random group size based on coin popularity
                group_size = np.random.choice(self.group_sizes[popularity])

                # Choose random mention rate based on activity level
                mention_rate = np.random.choice(self.mention_rates[activity])

                # Calculate mentions for this group
                group_mentions = int((group_size / 1000) * mention_rate)
                hour_mentions += group_mentions

            # Add some randomness and trends
            # Weekend effect
            if hour_time.weekday() >= 5:  # Saturday or Sunday
                hour_mentions *= np.random.uniform(0.8, 1.2)

            # Market events effect (random spikes)
            if np.random.random() < 0.05:  # 5% chance of a significant event
                hour_mentions *= np.random.uniform(1.5, 4.0)

            result[hour_key] = int(hour_mentions)

        # Cache the result
        self.cache[cache_key] = result
        self.cache_expiry[cache_key] = time.time() + self.cache_duration

        return result

    async def get_sentiment(self, coin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get estimated sentiment from Telegram for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary with sentiment information
        """
        # Check cache first
        cache_key = f"sentiment_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        # Get coin popularity
        popularity = self._get_coin_popularity(coin_name)

        # Get price trend to inform sentiment (in a real implementation, this would
        # be based on actual message content analysis)
        # For now, use random with bias based on popularity
        if popularity == 'high':
            base_sentiment = np.random.normal(0.6, 0.15)  # More positive for popular coins
        elif popularity == 'medium':
            base_sentiment = np.random.normal(0.5, 0.15)  # Neutral for medium coins
        else:
            base_sentiment = np.random.normal(0.4, 0.15)  # Slightly negative for less popular coins

        # Clamp to valid range
        sentiment_score = max(0.1, min(0.9, base_sentiment))

        # Determine sentiment category
        if sentiment_score > 0.7:
            sentiment = "very_positive"
        elif sentiment_score > 0.55:
            sentiment = "positive"
        elif sentiment_score > 0.45:
            sentiment = "neutral"
        elif sentiment_score > 0.3:
            sentiment = "negative"
        else:
            sentiment = "very_negative"

        result = {
            "sentiment": sentiment,
            "score": sentiment_score,
            "sample_size": self._estimate_group_count(coin_name, popularity) * 10,  # Rough estimate
            "source": "telegram_estimated"
        }

        # Cache the result
        self.cache[cache_key] = result
        self.cache_expiry[cache_key] = time.time() + self.cache_duration

        return result

    def _categorize_coins(self) -> Dict[str, str]:
        """
        Categorize coins by popularity

        :return: Dictionary mapping coin symbols to popularity categories
        """
        # Define popularity categories
        high_popularity = {
            'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'AVAX', 'SHIB'
        }

        medium_popularity = {
            'DOT', 'LINK', 'MATIC', 'UNI', 'LTC', 'TON', 'XMR', 'NEAR', 'XLM', 'ATOM',
            'FTM', 'ALGO', 'VET', 'HBAR', 'FIL', 'ICP', 'AAVE', 'SAND', 'MANA'
        }

        categories = {}

        # Assign categories
        for coin in high_popularity:
            categories[coin] = 'high'

        for coin in medium_popularity:
            categories[coin] = 'medium'

        return categories

    def _get_coin_popularity(self, coin_name: str) -> str:
        """
        Get popularity category for a coin

        :param coin_name: Name of the cryptocurrency
        :return: Popularity category ('high', 'medium', or 'low')
        """
        return self.coin_categories.get(coin_name.upper(), 'low')

    def _estimate_group_count(self, coin_name: str, popularity: str) -> int:
        """
        Estimate number of Telegram groups discussing a coin

        :param coin_name: Name of the cryptocurrency
        :param popularity: Popularity category
        :return: Estimated number of groups
        """
        if popularity == 'high':
            return np.random.randint(50, 200)
        elif popularity == 'medium':
            return np.random.randint(20, 50)
        else:
            return np.random.randint(5, 20)

    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("Telegram adapter cache cleared")
