import os
import logging
import crypto_detector.config.settings as settings
from typing import Dict, List, Optional, Any, Type

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter
from crypto_detector.adapters.twitter_social_adapter import TwitterAdapter
from crypto_detector.adapters.reddit_social_adapter import RedditAdapter
from crypto_detector.adapters.telegram_social_adapter import TelegramAdapter
from crypto_detector.analysis.social_analyzer import SocialAnalyzer

# Setup logging
logger = logging.getLogger(__name__)


class SocialMediaManager:
    """
    Class for managing social media adapters and analyzers.
    Provides a unified interface for interacting with all social media data.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the social media manager

        :param config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.adapters = {}
        self.analyzer = SocialAnalyzer()

        self.config = self._load_configuration()

        # Initialize adapters
        self._initialize_adapters()

    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from file

        :return: Configuration dictionary
        """
        config = {
            'twitter': {
                "api_key": settings.TWITTER_API_KEY,
                "api_secret": settings.TWITTER_API_SECRET,
                "bearer_token": settings.TWITTER_BEARER_TOKEN,
            },
            'reddit': {
                "client_id": settings.REDDIT_CLIENT_ID,
                "client_secret": settings.REDDIT_CLIENT_SECRET,
                "user_agent": settings.REDDIT_USER_AGENT,
            },
            'telegram': {
                "api_id": settings.TELEGRAM_API_ID,
                "api_hash": settings.TELEGRAM_API_HASH,
            },
            'general': {
                'rate_limit_delay': 1.0,
                'cache_duration': 300
            }
        }

        return config

    def _initialize_adapters(self):
        """Initialize all available social media adapters"""
        # Initialize Twitter adapter
        if 'api_key' in self.config['twitter'] or 'bearer_token' in self.config['twitter']:
            try:
                twitter_adapter = TwitterAdapter(
                    api_key=self.config['twitter'].get('api_key'),
                    api_secret=self.config['twitter'].get('api_secret'),
                    bearer_token=self.config['twitter'].get('bearer_token'),
                    cache_duration=self.config['twitter'].get('cache_duration',
                                                              self.config['general'].get('cache_duration', 300))
                )

                # Check if adapter can be used (has valid token)
                if twitter_adapter.bearer_token:
                    self.adapters['twitter'] = twitter_adapter
                    logger.info("Twitter adapter initialized successfully")
                else:
                    logger.warning("Twitter adapter initialized but no bearer token provided")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter adapter: {str(e)}")
        else:
            logger.info("No Twitter credentials provided, skipping Twitter adapter initialization")

        # Initialize Reddit adapter
        if 'client_id' in self.config['reddit'] or 'client_secret' in self.config['reddit']:
            try:
                reddit_adapter = RedditAdapter(
                    client_id=self.config['reddit'].get('client_id'),
                    client_secret=self.config['reddit'].get('client_secret'),
                    user_agent=self.config['reddit'].get('user_agent', 'crypto_detector/1.0'),
                    cache_duration=self.config['reddit'].get('cache_duration',
                                                             self.config['general'].get('cache_duration', 300))
                )

                # Check if adapter can be used (has valid credentials)
                if reddit_adapter.client_id and reddit_adapter.client_secret:
                    self.adapters['reddit'] = reddit_adapter
                    logger.info("Reddit adapter initialized successfully")
                else:
                    logger.warning("Reddit adapter initialized but no valid credentials provided")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit adapter: {str(e)}")
        else:
            logger.info("No Reddit credentials provided, skipping Reddit adapter initialization")

        # Initialize Telegram adapter (even in estimation mode)
        try:
            telegram_adapter = TelegramAdapter(
                api_id=self.config['telegram'].get('api_id'),
                api_hash=self.config['telegram'].get('api_hash'),
                cache_duration=self.config['telegram'].get('cache_duration',
                                                           self.config['general'].get('cache_duration', 300))
            )

            # Always add Telegram adapter (will work in estimation mode if no credentials)
            self.adapters['telegram'] = telegram_adapter

            if telegram_adapter.api_id and telegram_adapter.api_hash:
                logger.info("Telegram adapter initialized with actual credentials")
            else:
                logger.info("Telegram adapter initialized in estimation mode")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram adapter: {str(e)}")

        # Log summary
        active_platforms = list(self.adapters.keys())
        if active_platforms:
            logger.info(f"Active social media platforms: {', '.join(active_platforms)}")
        else:
            logger.warning("No social media adapters were initialized successfully")

    async def detect_social_media_mentions(self, symbol: str) -> Dict[str, Any]:
        """
        Detect social media mentions for a cryptocurrency across all platforms

        :param symbol: Cryptocurrency symbol (e.g., 'BTC/USDT')
        :return: Dictionary with analysis results
        """
        # Get active adapters
        active_adapters = list(self.adapters.values())

        if not active_adapters:
            logger.warning("No active social media adapters available")
            return {
                'social_signal': False,
                'mentions': 0,
                'average_mentions': 0,
                'percent_change': 0,
                'growth_acceleration': 0,
                'error': "No social media adapters available"
            }

        # Get analysis from the social analyzer
        result = await self.analyzer.detect_social_media_mentions(symbol, active_adapters)

        # Add platform-specific details
        platform_details = {}
        for platform, adapter in self.adapters.items():
            try:
                mentions = await adapter.get_current_mentions(coin_name=symbol.split('/')[0])
                platform_details[platform] = {
                    'mentions': mentions
                }

                # Add sentiment if available
                if hasattr(adapter, 'get_sentiment'):
                    sentiment = await adapter.get_sentiment(coin_name=symbol.split('/')[0])
                    if sentiment:
                        platform_details[platform]['sentiment'] = sentiment
            except Exception as e:
                logger.error(f"Error getting details from {platform}: {str(e)}")
                platform_details[platform] = {
                    'error': str(e)
                }

        # Add platform details to result
        result['platforms'] = platform_details

        return result

    async def get_historical_data(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical social media data for a cryptocurrency

        :param symbol: Cryptocurrency symbol (e.g., 'BTC/USDT')
        :param days: Number of days of historical data to retrieve
        :return: Dictionary with historical data
        """
        coin_name = symbol.split('/')[0]

        # Get data from each platform
        platform_data = {}
        for platform, adapter in self.adapters.items():
            try:
                historical_mentions = await adapter.get_historical_mentions(coin_name)
                platform_data[platform] = historical_mentions
            except Exception as e:
                logger.error(f"Error getting historical data from {platform}: {str(e)}")
                platform_data[platform] = {
                    'error': str(e)
                }

        # Get aggregated analysis
        volume_change = await self.analyzer.get_social_volume_change(coin_name)

        return {
            'symbol': symbol,
            'platform_data': platform_data,
            'volume_change': volume_change,
            'history': self.analyzer.get_coin_social_history(coin_name)
        }

    async def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a cryptocurrency across all platforms

        :param symbol: Cryptocurrency symbol (e.g., 'BTC/USDT')
        :return: Dictionary with sentiment analysis
        """
        coin_name = symbol.split('/')[0]

        # Get sentiment from each adapter
        sentiments = {}
        for platform, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'get_sentiment'):
                    sentiment = await adapter.get_sentiment(coin_name)
                    if sentiment:
                        sentiments[platform] = sentiment
            except Exception as e:
                logger.error(f"Error getting sentiment from {platform}: {str(e)}")

        # If we have no sentiment data, return error
        if not sentiments:
            return {
                'symbol': symbol,
                'error': "No sentiment data available"
            }

        # Calculate aggregate sentiment
        sentiment_scores = [s.get('score', 0.5) for s in sentiments.values() if 'score' in s]
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)

            # Determine overall sentiment
            if avg_score > 0.7:
                overall_sentiment = "very_positive"
            elif avg_score > 0.55:
                overall_sentiment = "positive"
            elif avg_score > 0.45:
                overall_sentiment = "neutral"
            elif avg_score > 0.3:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "very_negative"
        else:
            avg_score = 0.5
            overall_sentiment = "neutral"

        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'overall_score': avg_score,
            'platform_sentiments': sentiments
        }

    async def monitor_keywords(self, keywords: List[str], platforms: List[str] = None) -> Dict[str, Any]:
        """
        Monitor multiple keywords across social platforms

        :param keywords: List of keywords to monitor
        :param platforms: List of platforms to monitor (default: all available)
        :return: Dictionary with mention counts for each keyword
        """
        if platforms is None:
            platforms = list(self.adapters.keys())

        results = {}
        platform_results = {}

        for platform_name in platforms:
            if platform_name in self.adapters:
                adapter = self.adapters[platform_name]
                try:
                    platform_data = await self._monitor_keywords_on_platform(keywords, adapter)
                    platform_results[platform_name] = platform_data

                    # Aggregate results
                    for keyword, count in platform_data.items():
                        if keyword in results:
                            results[keyword] += count
                        else:
                            results[keyword] = count
                except Exception as e:
                    logger.error(f"Error monitoring keywords on {platform_name}: {str(e)}")
                    platform_results[platform_name] = {'error': str(e)}

        return {
            'aggregate_results': results,
            'platform_results': platform_results
        }

    async def _monitor_keywords_on_platform(self, keywords: List[str], adapter: SocialMediaAdapter) -> Dict[str, int]:
        """
        Monitor keywords on a specific platform

        :param keywords: List of keywords to monitor
        :param adapter: Social media adapter to use
        :return: Dictionary with mention counts for each keyword
        """
        results = {}

        for keyword in keywords:
            try:
                mentions = await adapter.get_current_mentions(keyword)
                results[keyword] = mentions
            except Exception as e:
                logger.error(f"Error monitoring keyword '{keyword}': {str(e)}")
                results[keyword] = 0

        return results

    def clear_caches(self):
        """Clear all adapter caches and analyzer history"""
        # Clear adapter caches
        for adapter in self.adapters.values():
            if hasattr(adapter, 'clear_cache'):
                adapter.clear_cache()

        # Clear analyzer history
        self.analyzer.clear_history()

        logger.info("All social media caches and history cleared")

    def get_active_platforms(self) -> List[str]:
        """
        Get list of active social media platforms

        :return: List of active platform names
        """
        return list(self.adapters.keys())

    def get_coin_history(self, symbol: str) -> List[int]:
        """
        Get stored history for a coin

        :param symbol: Cryptocurrency symbol
        :return: List of historical mention counts
        """
        coin_name = symbol.split('/')[0]
        return self.analyzer.get_coin_social_history(coin_name)
