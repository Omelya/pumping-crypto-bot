import asyncio
import os
import random
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter

# Setup logging
logger = logging.getLogger(__name__)


class TwitterAdapter(SocialMediaAdapter):
    """
    Adapter for fetching cryptocurrency mentions from Twitter.
    Uses Twitter API v2 with bearer token authentication.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 bearer_token: Optional[str] = None, cache_duration: int = 300):
        """
        Initialize the Twitter adapter

        :param api_key: Twitter API key (optional, will use env var TWITTER_API_KEY if not provided)
        :param api_secret: Twitter API secret (optional, will use env var TWITTER_API_SECRET if not provided)
        :param bearer_token: Twitter bearer token (optional, will use env var TWITTER_BEARER_TOKEN if not provided)
        :param cache_duration: How long to cache results in seconds (default: 5 minutes)
        """
        self.api_key = api_key or os.getenv('TWITTER_API_KEY', '')
        self.api_secret = api_secret or os.getenv('TWITTER_API_SECRET', '')
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN', '')
        self.cache_duration = cache_duration

        # Initialize cache
        self.cache = {}
        self.cache_expiry = {}

        if not self.bearer_token:
            logger.warning("No Twitter bearer token provided. Twitter adapter will be limited.")

    def name(self) -> str:
        """Get the name of this social media adapter"""
        return "Twitter"

    # Додаємо функцію для обробки запитів з повторними спробами
    async def _make_twitter_request(self, url: str, params: Dict, headers: Dict, max_retries: int = 3) -> Tuple[
        int, Any]:
        """
        Make a Twitter API request with retry logic for rate limits

        :param url: API endpoint URL
        :param params: Query parameters
        :param headers: Request headers
        :param max_retries: Maximum number of retries
        :return: Tuple of (status_code, response_data)
        """
        retry_count = 0
        backoff_time = 2  # Starting backoff time in seconds

        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        status = response.status
                        response_text = await response.text()

                        # If successful, return the response
                        if status == 200:
                            return status, await response.json()

                        # If rate limited, apply exponential backoff
                        if status == 429:
                            # Get retry-after header if available
                            retry_after = response.headers.get('retry-after')
                            if retry_after:
                                wait_time = int(retry_after)
                            else:
                                # Exponential backoff with jitter
                                wait_time = backoff_time + random.uniform(0, 1)
                                backoff_time *= 2

                            logger.warning(f"Twitter API rate limited. Waiting {wait_time} seconds before retry.")
                            await asyncio.sleep(wait_time)
                            retry_count += 1
                            continue

                        # Other error, log and return
                        return status, response_text
            except Exception as e:
                logger.error(f"Error making Twitter API request: {str(e)}")
                retry_count += 1
                await asyncio.sleep(backoff_time)
                backoff_time *= 2

        # If we've exhausted retries
        return 0, "Max retries reached"

    # Модифікований метод get_current_mentions з використанням нової функції
    async def get_current_mentions(self, coin_name: str) -> int:
        """
        Get the current number of mentions for a specific cryptocurrency on Twitter

        :param coin_name: Name of the cryptocurrency
        :return: Number of mentions
        """
        if not self.bearer_token:
            return 0

        # Check cache first
        cache_key = f"current_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Create search terms
            search_terms = self._create_search_terms(coin_name)
            query = " OR ".join([f'"{term}"' for term in search_terms])

            # Використовуємо UTC час для запобігання проблем з часовими поясами
            # Встановлюємо end_time на 30 секунд у минулому
            # (Twitter вимагає мінімум 10 секунд, але ми даємо більший запас)
            end_time = datetime.utcnow() - timedelta(seconds=30)
            start_time = end_time - timedelta(hours=1)

            # Формат RFC 3339 з Z на кінці для позначення UTC
            end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Додаткова перевірка для впевненості, що часові мітки в минулому
            now = datetime.utcnow()
            if end_time >= now or start_time >= now:
                logger.warning(f"Time calculation error: end_time or start_time is not in the past")
                end_time = now - timedelta(seconds=30)
                start_time = end_time - timedelta(hours=1)
                end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Twitter API v2 endpoint for recent search
            url = "https://api.twitter.com/2/tweets/counts/recent"
            params = {
                "query": query,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "granularity": "hour"
            }

            headers = {
                "Authorization": f"Bearer {self.bearer_token}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            mentions = data['data'][0]['tweet_count']
                            # Cache the result
                            self.cache[cache_key] = mentions
                            self.cache_expiry[cache_key] = time.time() + self.cache_duration
                            return mentions
                    else:
                        logger.warning(f"Twitter API error: {response.status} - {await response.text()}")

            # If API failed, return 0
            return 0

        except Exception as e:
            logger.error(f"Error fetching Twitter mentions for {coin_name}: {str(e)}")
            return 0

    async def get_historical_mentions(self, coin_name: str) -> Dict[str, int]:
        """
        Get historical Twitter mentions for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary mapping datetime strings to mention counts
        """
        if not self.bearer_token:
            return {}

        # Check cache first
        cache_key = f"historical_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Create search terms
            search_terms = self._create_search_terms(coin_name)
            query = " OR ".join([f'"{term}"' for term in search_terms])

            # Calculate time range (last 7 days)
            end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            start_time = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Twitter API v2 endpoint for recent search
            url = "https://api.twitter.com/2/tweets/counts/recent"
            params = {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "granularity": "hour"
            }

            headers = {
                "Authorization": f"Bearer {self.bearer_token}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = {}

                        if 'data' in data:
                            for hour_data in data['data']:
                                # Convert Twitter format to our datetime string format
                                dt = datetime.strptime(hour_data['end'], "%Y-%m-%dT%H:%M:%SZ")
                                hour_key = dt.strftime("%Y-%m-%d %H")
                                result[hour_key] = hour_data['tweet_count']

                            # Cache the result
                            self.cache[cache_key] = result
                            self.cache_expiry[cache_key] = time.time() + self.cache_duration
                            return result
                    else:
                        logger.warning(f"Twitter historical API error: {response.status} - {await response.text()}")

            # If API failed, return empty dict
            return {}

        except Exception as e:
            logger.error(f"Error fetching Twitter historical data for {coin_name}: {str(e)}")
            return {}

    async def get_sentiment(self, coin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis from Twitter for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary with sentiment information or None if not supported
        """
        if not self.bearer_token:
            return None

        # Check cache first
        cache_key = f"sentiment_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Create search terms
            search_terms = self._create_search_terms(coin_name)
            query = " OR ".join([f'"{term}"' for term in search_terms])

            # Calculate time range (last 24 hours)
            end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            start_time = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Twitter API v2 endpoint for recent tweets (not just counts)
            url = "https://api.twitter.com/2/tweets/search/recent"
            params = {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "tweet.fields": "public_metrics,lang",
                "max_results": 100
            }

            headers = {
                "Authorization": f"Bearer {self.bearer_token}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'data' in data and len(data['data']) > 0:
                            # Very simple sentiment analysis based on likes/retweets ratio
                            positive_count = 0
                            negative_count = 0
                            neutral_count = 0

                            for tweet in data['data']:
                                metrics = tweet.get('public_metrics', {})
                                likes = metrics.get('like_count', 0)
                                retweets = metrics.get('retweet_count', 0)
                                replies = metrics.get('reply_count', 0)

                                # Simple heuristic: tweets with more likes than retweets tend to be positive
                                if likes > (retweets * 2) and likes > 5:
                                    positive_count += 1
                                elif retweets > likes and replies > (likes / 2):
                                    negative_count += 1
                                else:
                                    neutral_count += 1

                            total = positive_count + negative_count + neutral_count
                            if total > 0:
                                positive_ratio = positive_count / total
                                negative_ratio = negative_count / total

                                # Determine sentiment
                                if positive_ratio > 0.6:
                                    sentiment = "positive"
                                    score = 0.6 + (positive_ratio - 0.6) * 0.5  # Scale to 0.6-0.85
                                elif negative_ratio > 0.6:
                                    sentiment = "negative"
                                    score = 0.4 - (negative_ratio - 0.6) * 0.5  # Scale to 0.15-0.4
                                else:
                                    sentiment = "neutral"
                                    score = 0.4 + (positive_ratio * 0.2)  # Scale to 0.4-0.6

                                result = {
                                    "sentiment": sentiment,
                                    "score": score,
                                    "sample_size": total,
                                    "source": "twitter"
                                }

                                # Cache the result
                                self.cache[cache_key] = result
                                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                                return result
                    else:
                        logger.warning(f"Twitter API error for sentiment: {response.status} - {await response.text()}")

        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment for {coin_name}: {str(e)}")

        return None

    def _create_search_terms(self, coin_name: str) -> List[str]:
        """
        Create search terms for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: List of search terms
        """
        search_terms = [
            f"#{coin_name.lower()}",
            f"#{coin_name.upper()}",
            f"#{coin_name.lower()}crypto",
            f"#{coin_name.upper()}crypto",
            coin_name.upper()
        ]

        # For Bitcoin, Ethereum add special terms
        if coin_name.upper() == "BTC":
            search_terms.extend(["#Bitcoin", "Bitcoin"])
        elif coin_name.upper() == "ETH":
            search_terms.extend(["#Ethereum", "Ethereum"])

        return search_terms

    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("Twitter adapter cache cleared")
