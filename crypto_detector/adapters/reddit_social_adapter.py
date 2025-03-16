import asyncio
import os
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from crypto_detector.adapters.social_media_adapter import SocialMediaAdapter

# Setup logging
logger = logging.getLogger(__name__)


class RedditAdapter(SocialMediaAdapter):
    """
    Adapter for fetching cryptocurrency mentions from Reddit.
    Uses Reddit API with OAuth2 authentication.
    """

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None,
                 user_agent: Optional[str] = None, cache_duration: int = 300):
        """
        Initialize the Reddit adapter

        :param client_id: Reddit API client ID (optional, will use env var REDDIT_CLIENT_ID if not provided)
        :param client_secret: Reddit API client secret (optional, will use env var REDDIT_CLIENT_SECRET if not provided)
        :param user_agent: User agent to use for API requests (optional, will use env var REDDIT_USER_AGENT if not provided)
        :param cache_duration: How long to cache results in seconds (default: 5 minutes)
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID', '')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET', '')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'crypto_detector/1.0')
        self.cache_duration = cache_duration

        # Authentication token
        self.auth_token = None
        self.token_expiry = 0

        # Initialize cache
        self.cache = {}
        self.cache_expiry = {}

        if not self.client_id or not self.client_secret:
            logger.warning("No Reddit API credentials provided. Reddit adapter will be limited.")

    def name(self) -> str:
        """Get the name of this social media adapter"""
        return "Reddit"

    async def get_current_mentions(self, coin_name: str) -> int:
        """
        Get the current number of mentions for a specific cryptocurrency on Reddit

        :param coin_name: Name of the cryptocurrency
        :return: Number of mentions
        """
        if not self.client_id or not self.client_secret:
            return 0

        # Check cache first
        cache_key = f"current_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Ensure we have a valid auth token
            token = await self._get_auth_token()
            if not token:
                return 0

            # Get relevant subreddits for cryptocurrency
            subreddits = self._get_crypto_subreddits(coin_name)

            # Create search terms
            search_terms = self._create_search_terms(coin_name)

            # Calculate timestamp for posts in the last hour
            time_threshold = int(time.time()) - 3600  # Last hour

            # Set API headers
            api_headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent
            }

            total_mentions = 0

            # Search in each subreddit
            for subreddit in subreddits:
                try:
                    # Rate limit to avoid hitting Reddit API limits
                    await self._rate_limit()

                    # Get recent posts
                    posts_url = f"https://oauth.reddit.com/r/{subreddit}/new"
                    params = {
                        "limit": 100  # Maximum allowed by Reddit API
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.get(posts_url, headers=api_headers, params=params) as posts_response:
                            if posts_response.status == 200:
                                posts_data = await posts_response.json()

                                if 'data' in posts_data and 'children' in posts_data['data']:
                                    for post in posts_data['data']['children']:
                                        post_data = post['data']
                                        created_time = post_data.get('created_utc', 0)

                                        # Check if post is within the last hour
                                        if created_time >= time_threshold:
                                            # Check if any search term is in the title or selftext
                                            title = post_data.get('title', '').lower()
                                            selftext = post_data.get('selftext', '').lower()

                                            for term in search_terms:
                                                if term.lower() in title or term.lower() in selftext:
                                                    total_mentions += 1
                                                    break
                            elif posts_response.status == 404:
                                logger.warning(f"Reddit API error for r/{subreddit}: 404 - Subreddit not found")
                            else:
                                logger.warning(f"Reddit API error for r/{subreddit}: {posts_response.status}")
                except Exception as sub_e:
                    # Log and continue with next subreddit
                    logger.error(f"Error processing subreddit r/{subreddit}: {str(sub_e)}")
                    continue

            # Cache the result
            self.cache[cache_key] = total_mentions
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            return total_mentions

        except Exception as e:
            logger.error(f"Error fetching Reddit mentions for {coin_name}: {str(e)}")
            return 0

    async def get_historical_mentions(self, coin_name: str) -> Dict[str, int]:
        """
        Get historical Reddit mentions for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary mapping datetime strings to mention counts
        """
        if not self.client_id or not self.client_secret:
            return {}

        # Check cache first
        cache_key = f"historical_mentions_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Ensure we have a valid auth token
            token = await self._get_auth_token()
            if not token:
                return {}

            # Create search terms
            search_terms = self._create_search_terms(coin_name)

            # Use Pushshift API for historical data (last 7 days)
            start_time = int((datetime.now() - timedelta(days=7)).timestamp())

            # Create query string
            query = " OR ".join(search_terms)

            result = {}

            # Use Pushshift for historical data
            pushshift_url = "https://api.pushshift.io/reddit/search/submission"
            params = {
                "q": query,
                "after": start_time,
                "size": 500,  # Maximum allowed
                "subreddit": ",".join(self._get_crypto_subreddits(coin_name, for_query=True))
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(pushshift_url, params=params) as ps_response:
                        if ps_response.status == 200:
                            posts_data = await ps_response.json()

                            if 'data' in posts_data:
                                for post in posts_data['data']:
                                    created_time = post.get('created_utc', 0)
                                    dt = datetime.fromtimestamp(created_time)
                                    hour_key = dt.strftime("%Y-%m-%d %H")

                                    if hour_key in result:
                                        result[hour_key] += 1
                                    else:
                                        result[hour_key] = 1
                        else:
                            logger.warning(f"Pushshift API error: {ps_response.status}")
            except Exception as ps_error:
                # Pushshift can be unreliable, log and continue
                logger.warning(f"Pushshift API error: {str(ps_error)}")

                # Fallback to Reddit API if Pushshift fails
                await self._fallback_to_reddit_api(coin_name, result, token)

            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            return result

        except Exception as e:
            logger.error(f"Error fetching Reddit historical data for {coin_name}: {str(e)}")
            return {}

    async def get_sentiment(self, coin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis from Reddit for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: Dictionary with sentiment information or None if not supported
        """
        if not self.client_id or not self.client_secret:
            return None

        # Check cache first
        cache_key = f"sentiment_{coin_name}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]

        try:
            # Ensure we have a valid auth token
            token = await self._get_auth_token()
            if not token:
                return None

            # Get crypto subreddits
            subreddits = self._get_crypto_subreddits(coin_name)

            # Create search terms
            search_terms = self._create_search_terms(coin_name)

            # Set API headers
            api_headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self.user_agent
            }

            # Track sentiment metrics
            positive_score = 0
            negative_score = 0
            neutral_score = 0
            total_posts = 0

            # Search in each subreddit
            for subreddit in subreddits[:3]:  # Limit to top 3 subreddits to avoid rate limits
                # Rate limit to avoid hitting Reddit API limits
                await self._rate_limit()

                # Get recent posts
                posts_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                params = {
                    "q": coin_name.upper(),
                    "sort": "relevance",
                    "t": "week",
                    "limit": 25  # Limit to 25 posts per subreddit
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(posts_url, headers=api_headers, params=params) as posts_response:
                        if posts_response.status == 200:
                            posts_data = await posts_response.json()

                            if 'data' in posts_data and 'children' in posts_data['data']:
                                for post in posts_data['data']['children']:
                                    post_data = post['data']

                                    # Use upvote ratio and score as sentiment indicators
                                    upvote_ratio = post_data.get('upvote_ratio', 0.5)
                                    score = post_data.get('score', 0)

                                    # Count post in totals
                                    total_posts += 1

                                    # Analyze sentiment based on upvote ratio and score
                                    if upvote_ratio > 0.75 and score > 10:
                                        positive_score += 1
                                    elif upvote_ratio < 0.4 or score < 0:
                                        negative_score += 1
                                    else:
                                        neutral_score += 1
                        else:
                            logger.warning(f"Reddit API error for sentiment: {posts_response.status}")

            # Calculate overall sentiment if we have data
            if total_posts > 0:
                # Calculate ratios
                positive_ratio = positive_score / total_posts
                negative_ratio = negative_score / total_posts

                # Determine sentiment
                if positive_ratio > 0.6:
                    sentiment = "positive"
                    score = 0.6 + (positive_ratio - 0.6) * 0.5  # Scale to 0.6-0.85
                elif negative_ratio > 0.4:
                    sentiment = "negative"
                    score = 0.4 - (negative_ratio - 0.4) * 0.5  # Scale to 0.2-0.4
                else:
                    sentiment = "neutral"
                    score = 0.4 + (positive_ratio * 0.2)  # Scale to 0.4-0.6

                result = {
                    "sentiment": sentiment,
                    "score": score,
                    "sample_size": total_posts,
                    "source": "reddit"
                }

                # Cache the result
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                return result

        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {coin_name}: {str(e)}")

        return None

    async def _get_auth_token(self) -> Optional[str]:
        """
        Get an authentication token from Reddit API

        :return: Authentication token or None if error
        """
        # Check if we already have a valid token
        if self.auth_token and time.time() < self.token_expiry:
            return self.auth_token

        try:
            # Reddit OAuth2 authentication
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            auth_headers = {
                "User-Agent": self.user_agent
            }

            async with aiohttp.ClientSession() as session:
                # Get access token
                async with session.post(
                        auth_url,
                        data=auth_data,
                        headers=auth_headers,
                        auth=aiohttp.BasicAuth(self.client_id, self.client_secret)
                ) as auth_response:

                    if auth_response.status != 200:
                        logger.warning(f"Reddit auth error: {auth_response.status} - {await auth_response.text()}")
                        return None

                    auth_data = await auth_response.json()
                    token = auth_data.get('access_token')
                    expires_in = auth_data.get('expires_in', 3600)

                    if not token:
                        logger.warning("Reddit auth failed: No token received")
                        return None

                    # Save token and expiry
                    self.auth_token = token
                    self.token_expiry = time.time() + expires_in - 60  # Subtract 60 seconds for safety
                    return token

        except Exception as e:
            logger.error(f"Error getting Reddit auth token: {str(e)}")
            return None

    def _create_search_terms(self, coin_name: str) -> List[str]:
        """
        Create search terms for a cryptocurrency

        :param coin_name: Name of the cryptocurrency
        :return: List of search terms
        """
        search_terms = [
            coin_name.upper(),
            coin_name.lower(),
        ]

        # For Bitcoin, Ethereum add special terms
        if coin_name.upper() == "BTC":
            search_terms.extend(["Bitcoin", "bitcoin"])
        elif coin_name.upper() == "ETH":
            search_terms.extend(["Ethereum", "ethereum"])

        return search_terms

    async def _rate_limit(self, delay: float = 1.0):
        """
        Apply rate limiting to avoid hitting API limits

        :param delay: Delay in seconds
        """
        await asyncio.sleep(delay)

    async def _fallback_to_reddit_api(self, coin_name: str, result_dict: Dict[str, int], token: str):
        """
        Fallback to Reddit API if Pushshift fails

        :param coin_name: Name of the cryptocurrency
        :param result_dict: Dictionary to populate with results
        :param token: Authentication token
        """
        # Get relevant subreddits for cryptocurrency
        subreddits = self._get_crypto_subreddits(coin_name)[:3]  # Limit to top 3

        # Set API headers
        api_headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.user_agent
        }

        for subreddit in subreddits:
            # Rate limit
            await self._rate_limit()

            # Get posts from the past week
            posts_url = f"https://oauth.reddit.com/r/{subreddit}/new"
            params = {
                "limit": 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(posts_url, headers=api_headers, params=params) as posts_response:
                    if posts_response.status == 200:
                        posts_data = await posts_response.json()

                        if 'data' in posts_data and 'children' in posts_data['data']:
                            search_terms = self._create_search_terms(coin_name)

                            for post in posts_data['data']['children']:
                                post_data = post['data']
                                created_time = post_data.get('created_utc', 0)
                                dt = datetime.fromtimestamp(created_time)
                                hour_key = dt.strftime("%Y-%m-%d %H")

                                # Check if any search term is in the title or selftext
                                title = post_data.get('title', '').lower()
                                selftext = post_data.get('selftext', '').lower()

                                for term in search_terms:
                                    if term.lower() in title or term.lower() in selftext:
                                        if hour_key in result_dict:
                                            result_dict[hour_key] += 1
                                        else:
                                            result_dict[hour_key] = 1
                                        break

    def _get_crypto_subreddits(self, coin_name: str, for_query: bool = False) -> List[str]:
        """
        Get relevant cryptocurrency subreddits

        :param coin_name: Cryptocurrency name
        :param for_query: Whether this is for a query string (comma-separated) or a list
        :return: List of subreddit names
        """
        # Покращений список основних криптовалютних субреддітів
        # Перевірені існуючі субреддіти
        subreddits = [
            "cryptocurrency",
            "cryptomarkets",
            "bitcoin",
            "altcoin",
            "binance",
            "cryptomoonshots",
            "satoshistreetbets"
        ]

        # Додавання спеціальних субреддітів для популярних монет
        coin_specific_subreddits = {
            "BTC": ["bitcoin", "bitcoinmarkets"],
            "ETH": ["ethereum", "ethtrader", "ethfinance"],
            "ADA": ["cardano"],
            "DOGE": ["dogecoin"],
            "XRP": ["ripple"],
            "SOL": ["solana"],
            "LINK": ["chainlink"],
            "DOT": ["dot", "polkadot"],
            "AVAX": ["avalancheavax"],
            "MATIC": ["0xpolygon", "maticnetwork"],
            "BNB": ["binance"],
            "SHIB": ["shibarmy"],
            "PEPE": ["pepecoins"],
            "ATOM": ["cosmosnetwork"],
            "NEAR": ["nearprotocol"],
            "FTM": ["fantomfoundation"],
            "LTC": ["litecoin"],
            "XMR": ["monero"],
            "ALGO": ["algorand"]
        }

        # Додавання специфічних для монети субреддітів, якщо вони існують
        if coin_name.upper() in coin_specific_subreddits:
            subreddits.extend(coin_specific_subreddits[coin_name.upper()])
        elif len(
                coin_name) >= 3:  # Додаємо монету як субреддіт тільки якщо назва достатньо довга (щоб уникнути таких як a8)
            # Перевіряємо тільки базові монети, а не короткі символи
            # Це запобігає спробам отримати доступ до неіснуючих субреддітів
            subreddits.append(coin_name.lower())

        # Видалення дублікатів, зберігаючи порядок
        unique_subreddits = []
        for subreddit in subreddits:
            if subreddit not in unique_subreddits:
                unique_subreddits.append(subreddit)

        return unique_subreddits

    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.cache_expiry = {}
        logger.info("Reddit adapter cache cleared")
