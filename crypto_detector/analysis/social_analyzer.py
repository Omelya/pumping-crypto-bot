import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any
import logging

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter

# Setup logging
logger = logging.getLogger(__name__)


class SocialAnalyzer:
    """
    Class for analyzing social media activity around cryptocurrencies.
    Tracks mentions on social networks and detects anomalous spikes in activity.

    This class focuses on analysis and delegates data collection to social media adapters.
    """

    def __init__(self):
        """
        Initialize the social media activity analyzer
        """
        self.social_history = {}  # History of mentions for different coins
        self.last_analysis = {}  # Store the last analysis results for each coin

    async def detect_social_media_mentions(
            self,
            symbol: str,
            social_adapters: List["SocialMediaAdapter"],
    ) -> Dict[str, Any]:
        """
        Analyze social media mentions

        :param symbol: Cryptocurrency symbol
        :param social_adapters: List of social media adapters to gather data from
        :return: Dict with analysis results
        """
        # Extract coin name from symbol
        coin_name = symbol.split('/')[0]

        # Save mention history if not already collected
        if coin_name not in self.social_history:
            self.social_history[coin_name] = deque(maxlen=24)  # Store data for the last 24 hours

            # Initialize with historical data
            historical_mentions = await self._gather_historical_mentions(coin_name, social_adapters)
            for mentions in historical_mentions:
                self.social_history[coin_name].append(mentions)

        # Get current mentions from social media platforms
        current_mentions = await self._gather_current_mentions(coin_name, social_adapters)

        # Get sentiment data if available
        sentiment_data = await self._gather_sentiment(coin_name, social_adapters)

        # Add to history
        self.social_history[coin_name].append(current_mentions)

        # Calculate statistics
        avg_mentions = np.mean(list(self.social_history[coin_name])[:-1]) if len(
            self.social_history[coin_name]) > 1 else current_mentions

        # Calculate change
        percent_change = ((current_mentions / avg_mentions) - 1) * 100 if avg_mentions > 0 else 0

        # Analyze growth acceleration
        mentions_history = list(self.social_history[coin_name])
        if len(mentions_history) >= 6:
            recent_growth_rate = (mentions_history[-1] / mentions_history[-2] - 1) if mentions_history[-2] > 0 else 0
            earlier_growth_rate = (mentions_history[-3] / mentions_history[-4] - 1) if mentions_history[-4] > 0 else 0
            growth_acceleration = recent_growth_rate - earlier_growth_rate
        else:
            growth_acceleration = 0

        # Create result
        result = {
            'social_signal': percent_change > 40 or growth_acceleration > 0.5,
            'mentions': current_mentions,
            'average_mentions': avg_mentions,
            'percent_change': percent_change,
            'growth_acceleration': growth_acceleration,
            'sentiment': sentiment_data
        }

        # Store this analysis
        self.last_analysis[coin_name] = result

        return result

    async def _gather_current_mentions(self, coin_name: str, social_adapters: List["SocialMediaAdapter"]) -> int:
        """
        Gather current mentions from all social media platforms

        :param coin_name: Name of the cryptocurrency
        :param social_adapters: List of social media adapters to gather data from
        :return: Total number of mentions
        """
        total_mentions = 0

        # Collect mentions from all adapters
        for adapter in social_adapters:
            try:
                adapter_mentions = await adapter.get_current_mentions(coin_name)
                total_mentions += adapter_mentions
            except Exception as e:
                logger.error(f"Error getting mentions from {adapter.__class__.__name__}: {str(e)}")

        return total_mentions

    async def _gather_historical_mentions(self, coin_name: str, social_adapters: List["SocialMediaAdapter"]) -> List[
        int]:
        """
        Gather historical mentions from all social media platforms

        :param coin_name: Name of the cryptocurrency
        :param social_adapters: List of social media adapters to gather data from
        :return: List of historical mention counts
        """
        historical_data = []
        aggregated_hourly_data = {}

        # Collect historical data from all adapters
        for adapter in social_adapters:
            try:
                adapter_history = await adapter.get_historical_mentions(coin_name)

                # Merge the data into our aggregated dictionary
                for hour_key, count in adapter_history.items():
                    if hour_key in aggregated_hourly_data:
                        aggregated_hourly_data[hour_key] += count
                    else:
                        aggregated_hourly_data[hour_key] = count
            except Exception as e:
                logger.error(f"Error getting historical mentions from {adapter.__class__.__name__}: {str(e)}")

        # Convert the aggregated dictionary into a timeseries list
        if aggregated_hourly_data:
            # Sort by hour keys
            sorted_hours = sorted(aggregated_hourly_data.keys())
            hourly_data = [aggregated_hourly_data[hour] for hour in sorted_hours]

            # Return the hourly data
            return hourly_data

        # If no historical data could be collected, create a reasonable baseline
        # Generate baseline based on coin popularity
        popular_coins = {'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'AVAX'}
        medium_coins = {'DOT', 'SHIB', 'LINK', 'MATIC', 'UNI', 'LTC', 'TON', 'XMR'}

        if coin_name in popular_coins:
            baseline = np.random.normal(1000, 200, 23)
        elif coin_name in medium_coins:
            baseline = np.random.normal(500, 100, 23)
        else:
            baseline = np.random.normal(100, 30, 23)

        return [int(m) for m in baseline]

    async def _gather_sentiment(self, coin_name: str, social_adapters: List["SocialMediaAdapter"]) -> Dict[str, Any]:
        """
        Gather sentiment analysis from social media platforms

        :param coin_name: Name of the cryptocurrency
        :param social_adapters: List of social media adapters to gather data from
        :return: Dictionary with sentiment analysis results
        """
        # Get sentiment data from adapters that support it
        sentiment_scores = []
        sentiment_labels = []

        for adapter in social_adapters:
            try:
                if hasattr(adapter, 'get_sentiment'):
                    sentiment = await adapter.get_sentiment(coin_name)
                    if sentiment:
                        sentiment_scores.append(sentiment.get('score', 0.5))
                        sentiment_labels.append(sentiment.get('sentiment', 'neutral'))
            except Exception as e:
                logger.error(f"Error getting sentiment from {adapter.__class__.__name__}: {str(e)}")

        # If we have sentiment data, average it
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)

            # Determine overall sentiment label based on average score
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

            return {
                "sentiment": overall_sentiment,
                "score": avg_score,
                "sources": len(sentiment_scores)
            }

        # Fallback: return basic neutral sentiment
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "sources": 0
        }

    def get_coin_social_history(self, coin_name: str) -> List[int]:
        """
        Get history of mentions for a specific coin

        :param coin_name: Name of the coin
        :return: List of mention counts or empty list if no history
        """
        return list(self.social_history.get(coin_name, []))

    def clear_history(self):
        """
        Clear social activity history
        """
        self.social_history = {}
        self.last_analysis = {}
        logger.info("Social media history cleared")

    async def get_social_volume_change(self, coin_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get detailed social volume changes over time

        :param coin_name: Name of the cryptocurrency
        :param hours: Number of hours to analyze
        :return: Dictionary with social volume change analysis
        """
        # We need history to perform this analysis
        history = self.get_coin_social_history(coin_name)

        # We need at least two data points
        if len(history) < 2:
            return {
                "insufficient_data": True,
                "available_hours": len(history),
                "coin": coin_name
            }

        # Calculate changes for available history
        current = history[-1]
        oldest = history[0]

        # For periods with enough data
        hour1_change = None
        hour4_change = None
        hour12_change = None
        hour24_change = None

        if len(history) >= 2:
            hour1_change = ((current / history[-2]) - 1) * 100 if history[-2] > 0 else 0

        if len(history) >= 5:
            hour4_ago = history[-5]
            hour4_change = ((current / hour4_ago) - 1) * 100 if hour4_ago > 0 else 0

        if len(history) >= 13:
            hour12_ago = history[-13]
            hour12_change = ((current / hour12_ago) - 1) * 100 if hour12_ago > 0 else 0

        if len(history) >= 25:
            hour24_ago = history[-25]
            hour24_change = ((current / hour24_ago) - 1) * 100 if hour24_ago > 0 else 0

        # Calculate overall change
        overall_change = ((current / oldest) - 1) * 100 if oldest > 0 else 0

        # Calculate deviation from average
        avg = np.mean(history)
        std_dev = np.std(history)
        z_score = (current - avg) / std_dev if std_dev > 0 else 0

        # Determine if this is an anomaly
        is_anomaly = z_score > 2.0
        anomaly_level = None

        if z_score > 3.0:
            anomaly_level = "high"
        elif z_score > 2.0:
            anomaly_level = "medium"
        elif z_score > 1.5:
            anomaly_level = "low"

        return {
            "coin": coin_name,
            "current_mentions": current,
            "average_mentions": avg,
            "z_score": z_score,
            "is_anomaly": is_anomaly,
            "anomaly_level": anomaly_level,
            "changes": {
                "1h": hour1_change,
                "4h": hour4_change,
                "12h": hour12_change,
                "24h": hour24_change,
                "overall": overall_change
            },
            "history_hours": len(history)
        }
