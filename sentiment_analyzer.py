import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Union
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class SolanaMarketSentimentAnalyzer:
    """
    Personalized Trading Assistant - Sentiment Analysis Module
    
    Analyzes market sentiment from social media, news, and on-chain data
    to provide personalized insights and trading recommendations.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with configuration parameters"""
        self.sentiment_sources = {
            "twitter": 0.40,  # Weight for Twitter/X sentiment
            "reddit": 0.20,   # Weight for Reddit sentiment
            "discord": 0.15,  # Weight for Discord sentiment
            "news": 0.15,     # Weight for news sentiment
            "onchain": 0.10   # Weight for on-chain metrics sentiment
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            "very_negative": -0.6,
            "negative": -0.2,
            "neutral": 0.2,
            "positive": 0.6,
            "very_positive": 1.0
        }
        
        # Initialize sentiment cache
        self.sentiment_cache = {}
        self.topic_models = {}
        
    def analyze_token_sentiment(self, token_address: str, token_symbol: str = None) -> Dict:
        """
        Perform comprehensive sentiment analysis for a token
        
        Args:
            token_address: Solana token address
            token_symbol: Optional token symbol for better social media matching
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        # Get data from various sources
        social_data = self._get_social_media_data(token_address, token_symbol)
        news_data = self._get_news_data(token_address, token_symbol)
        onchain_data = self._get_onchain_metrics(token_address)
        
        # Calculate sentiment for each source
        twitter_sentiment = self._analyze_twitter_sentiment(social_data.get("twitter", []))
        reddit_sentiment = self._analyze_reddit_sentiment(social_data.get("reddit", []))
        discord_sentiment = self._analyze_discord_sentiment(social_data.get("discord", []))
        news_sentiment = self._analyze_news_sentiment(news_data)
        onchain_sentiment = self._analyze_onchain_sentiment(onchain_data)
        
        # Combine sentiments with weights
        overall_sentiment = (
            twitter_sentiment["score"] * self.sentiment_sources["twitter"] +
            reddit_sentiment["score"] * self.sentiment_sources["reddit"] +
            discord_sentiment["score"] * self.sentiment_sources["discord"] +
            news_sentiment["score"] * self.sentiment_sources["news"] +
            onchain_sentiment["score"] * self.sentiment_sources["onchain"]
        )
        
        # Identify key topics and trends
        key_topics = self._extract_key_topics(
            twitter_sentiment["texts"] + 
            reddit_sentiment["texts"] + 
            discord_sentiment["texts"] + 
            news_sentiment["texts"]
        )
        
        # Generate personalized insights
        insights = self._generate_insights(
            overall_sentiment, 
            {
                "twitter": twitter_sentiment,
                "reddit": reddit_sentiment,
                "discord": discord_sentiment,
                "news": news_sentiment,
                "onchain": onchain_sentiment
            },
            key_topics
        )
        
        # Cache sentiment results
        self.sentiment_cache[token_address] = {
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "sentiment_label": self._get_sentiment_label(overall_sentiment)
        }
        
        return {
            "token_address": token_address,
            "symbol": token_symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "sentiment_label": self._get_sentiment_label(overall_sentiment),
            "source_sentiment": {
                "twitter": twitter_sentiment["score"],
                "reddit": reddit_sentiment["score"],
                "discord": discord_sentiment["score"],
                "news": news_sentiment["score"],
                "onchain": onchain_sentiment["score"]
            },
            "key_topics": key_topics,
            "insights": insights,
            "recommendations": self._generate_recommendations(overall_sentiment, key_topics, onchain_data)
        }
    
    def _get_social_media_data(self, token_address: str, token_symbol: str = None) -> Dict[str, List[Dict]]:
        """
        Retrieve social media data related to the token
        
        Args:
            token_address: Token address
            token_symbol: Token symbol
            
        Returns:
            Dictionary of social media data by platform
        """
        # In production: Call APIs to get real social media data
        # Using simulated data for demonstration
        
        # Generate fake social media data
        search_terms = [token_address[:8]]
        if token_symbol:
            search_terms.extend([token_symbol, f"${token_symbol}"])
            
        # Set sentiment distribution based on random seed from token_address
        seed = sum(ord(c) for c in token_address[:8])
        np.random.seed(seed)
        
        # Base sentiment slightly positive or negative
        base_sentiment = np.random.uniform(-0.3, 0.3)
        
        # Generate Twitter data
        twitter_count = np.random.randint(50, 200)
        twitter_data = []
        
        # Simulate some common crypto Twitter phrases
        twitter_phrases = [
            "Just bought some {symbol}! To the moon! 🚀",
            "Selling my {symbol}, too much uncertainty right now",
            "{symbol} looking bullish, great chart setup",
            "Beware of {symbol}, potential rug",
            "Whales accumulating {symbol}, something big coming?",
            "{symbol} volume increasing, breaking out soon",
            "New partnership announcement for {symbol}!",
            "{symbol} is dead, moving on to next gem",
            "Diamond hands on {symbol}, not selling",
            "Just discovered {symbol}, looks promising"
        ]
        
        for _ in range(twitter_count):
            sentiment_noise = np.random.normal(0, 0.3)
            sentiment = np.clip(base_sentiment + sentiment_noise, -1, 1)
            
            # Select phrase based on sentiment
            if sentiment > 0.3:
                phrase_indices = [0, 2, 4, 5, 6, 8, 9]  # Positive phrases
            elif sentiment < -0.3:
                phrase_indices = [1, 3, 7]  # Negative phrases
            else:
                phrase_indices = range(len(twitter_phrases))
                
            phrase = twitter_phrases[np.random.choice(phrase_indices)]
            text = phrase.format(symbol=token_symbol if token_symbol else token_address[:8])
            
            twitter_data.append({
                "text": text,
                "timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 48))).isoformat(),
                "likes": np.random.randint(0, 1000),
                "retweets": np.random.randint(0, 200),
                "user_followers": np.random.randint(50, 100000),
                "_simulated_sentiment": sentiment  # For demonstration
            })
        
        # Generate Reddit data (similar approach)
        reddit_count = np.random.randint(20, 100)
        reddit_data = []
        
        for _ in range(reddit_count):
            sentiment_noise = np.random.normal(0, 0.4)  # More variance on Reddit
            sentiment = np.clip(base_sentiment + sentiment_noise, -1, 1)
            
            # Longer text for Reddit
            text_length = np.random.randint(1, 5)
            text = " ".join([
                "Lorem ipsum dolor sit amet" if i % 2 == 0 else f"Discussion about {token_symbol if token_symbol else token_address[:8]}"
                for i in range(text_length)
            ])
            
            reddit_data.append({
                "text": text,
                "timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
                "upvotes": np.random.randint(-10, 500),
                "comments": np.random.randint(0, 50),
                "subreddit": np.random.choice(["SolanaNFTs", "solana", "CryptoCurrency", "SolanaTrading"]),
                "_simulated_sentiment": sentiment  # For demonstration
            })
        
        # Generate Discord data (similar approach, shorter)
        discord_count = np.random.randint(10, 80)
        discord_data = []
        
        for _ in range(discord_count):
            sentiment_noise = np.random.normal(0, 0.5)  # Even more variance on Discord
            sentiment = np.clip(base_sentiment + sentiment_noise, -1, 1)
            
            discord_data.append({
                "text": f"Message about {token_symbol if token_symbol else token_address[:8]} with sentiment {sentiment:.2f}",
                "timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 24))).isoformat(),
                "channel": np.random.choice(["general", "trading", "price-discussion", "alpha"]),
                "server": np.random.choice(["Solana", "SolTraders", "DeFi", "CryptoSignals"]),
                "_simulated_sentiment": sentiment  # For demonstration
            })
            
        return {
            "twitter": twitter_data,
            "reddit": reddit_data,
            "discord": discord_data
        }
    
    def _get_news_data(self, token_address: str, token_symbol: str = None) -> List[Dict]:
        """
        Retrieve news articles related to the token
        
        Args:
            token_address: Token address
            token_symbol: Token symbol
            
        Returns:
            List of news article data
        """
        # In production: Call news API to get real articles
        # Using simulated data for demonstration
        
        # Set random seed based on token address
        seed = sum(ord(c) for c in token_address[:8]) + 1
        np.random.seed(seed)
        
        news_count = np.random.randint(3, 15)
        base_sentiment = np.random.uniform(-0.3, 0.3)
        
        news_data = []
        news_headlines = [
            "{symbol} sees surge in trading volume as market recovers",
            "Analysts remain bullish on {symbol} despite market downturn",
            "Is {symbol} the next 100x gem on Solana?",
            "Concerns grow over {symbol} tokenomics and distribution",
            "New security vulnerabilities discovered in {symbol} contract",
            "{symbol} launches new features, community excited",
            "Whale accumulation of {symbol} reaches all-time high",
            "{symbol} team announces roadmap for Q3",
            "Market manipulation allegations surround {symbol} price action",
            "{symbol} integration with major Solana protocols announced"
        ]
        
        for _ in range(news_count):
            sentiment_noise = np.random.normal(0, 0.2)  # News tends to be more measured
            sentiment = np.clip(base_sentiment + sentiment_noise, -1, 1)
            
            # Select headline based on sentiment
            if sentiment > 0.2:
                headline_indices = [0, 1, 2, 5, 6, 7, 9]  # Positive headlines
            elif sentiment < -0.2:
                headline_indices = [3, 4, 8]  # Negative headlines
            else:
                headline_indices = range(len(news_headlines))
                
            headline = news_headlines[np.random.choice(headline_indices)]
            headline = headline.format(symbol=token_symbol if token_symbol else token_address[:8])
            
            news_data.append({
                "headline": headline,
                "summary": f"News summary about {token_symbol if token_symbol else token_address[:8]}",
                "url": f"https://example.com/news/{np.random.randint(1000, 9999)}",
                "source": np.random.choice(["CoinDesk", "CryptoNews", "Blockworks", "The Block", "Decrypt"]),
                "published_at": (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
                "_simulated_sentiment": sentiment  # For demonstration
            })
            
        return news_data
    
    def _get_onchain_metrics(self, token_address: str) -> Dict:
        """
        Retrieve on-chain metrics for sentiment analysis
        
        Args:
            token_address: Token address
            
        Returns:
            Dictionary of on-chain metrics
        """
        # In production: Query blockchain for real metrics
        # Using simulated data for demonstration
        
        # Set random seed based on token address
        seed = sum(ord(c) for c in token_address[:8]) + 2
        np.random.seed(seed)
        
        return {
            "unique_holders": np.random.randint(100, 10000),
            "daily_active_addresses": np.random.randint(10, 1000),
            "transaction_count_24h": np.random.randint(50, 5000),
            "volume_24h": np.random.randint(10000, 10000000),
            "holder_concentration": np.random.random() * 100,  # Percentage held by top 10 wallets
            "new_holders_24h": np.random.randint(0, 100),
            "average_transaction_size": np.random.randint(100, 10000),
            "buy_sell_ratio": np.random.uniform(0.5, 1.5)  # Ratio of buys to sells
        }
    
    def _analyze_twitter_sentiment(self, tweets: List[Dict]) -> Dict:
        """
        Analyze sentiment from Twitter data
        
        Args:
            tweets: List of tweet data
            
        Returns:
            Twitter sentiment analysis results
        """
        if not tweets:
            return {"score": 0, "count": 0, "texts": []}
            
        # In production: Use NLP model for sentiment analysis
        # For demonstration, we use the simulated sentiment
        
        # Weight by engagement (likes, retweets) and follower count
        weighted_sentiments = []
        texts = []
        
        for tweet in tweets:
            # Calculate engagement score
            engagement = np.log1p(tweet.get("likes", 0) + 3 * tweet.get("retweets", 0))
            
            # Calculate influence from follower count (log scale)
            influence = np.log1p(tweet.get("user_followers", 0)) / 10
            
            # Combined weight (engagement + influence)
            weight = 1 + engagement * 0.5 + influence * 0.5
            
            # Get sentiment (using simulated value)
            sentiment = tweet.get("_simulated_sentiment", 0)
            
            weighted_sentiments.append((sentiment, weight))
            texts.append(tweet.get("text", ""))
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_sentiments)
        if total_weight > 0:
            avg_sentiment = sum(sentiment * weight for sentiment, weight in weighted_sentiments) / total_weight
        else:
            avg_sentiment = 0
            
        return {
            "score": avg_sentiment,
            "count": len(tweets),
            "texts": texts,
            "label": self._get_sentiment_label(avg_sentiment)
        }
    
    def _analyze_reddit_sentiment(self, posts: List[Dict]) -> Dict:
        """
        Analyze sentiment from Reddit data
        
        Args:
            posts: List of Reddit post data
            
        Returns:
            Reddit sentiment analysis results
        """
        if not posts:
            return {"score": 0, "count": 0, "texts": []}
            
        # Similar approach to Twitter sentiment, but with Reddit-specific weights
        weighted_sentiments = []
        texts = []
        
        for post in posts:
            # Calculate engagement (upvotes and comments)
            upvotes = post.get("upvotes", 0)
            comments = post.get("comments", 0)
            
            # Upvotes can be negative on Reddit
            engagement = np.log1p(abs(upvotes) + 2 * comments) * (1 if upvotes >= 0 else -0.5)
            
            # Weight based on engagement
            weight = 1 + max(0, engagement * 0.7)
            
            # Get sentiment (using simulated value)
            sentiment = post.get("_simulated_sentiment", 0)
            
            weighted_sentiments.append((sentiment, weight))
            texts.append(post.get("text", ""))
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_sentiments)
        if total_weight > 0:
            avg_sentiment = sum(sentiment * weight for sentiment, weight in weighted_sentiments) / total_weight
        else:
            avg_sentiment = 0
            
        return {
            "score": avg_sentiment,
            "count": len(posts),
            "texts": texts,
            "label": self._get_sentiment_label(avg_sentiment)
        }
    
    def _analyze_discord_sentiment(self, messages: List[Dict]) -> Dict:
        """
        Analyze sentiment from Discord data
        
        Args:
            messages: List of Discord message data
            
        Returns:
            Discord sentiment analysis results
        """
        if not messages:
            return {"score": 0, "count": 0, "texts": []}
            
        # For Discord, we just use a simple average as engagement metrics are limited
        sentiments = []
        texts = []
        
        for message in messages:
            # Get sentiment (using simulated value)
            sentiment = message.get("_simulated_sentiment", 0)
            sentiments.append(sentiment)
            texts.append(message.get("text", ""))
            
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
        return {
            "score": avg_sentiment,
            "count": len(messages),
            "texts": texts,
            "label": self._get_sentiment_label(avg_sentiment)
        }
    
    def _analyze_news_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment from news articles
        
        Args:
            articles: List of news article data
            
        Returns:
            News sentiment analysis results
        """
        if not articles:
            return {"score": 0, "count": 0, "texts": []}
            
        # For news, weight by recency
        weighted_sentiments = []
        texts = []
        
        now = datetime.now()
        
        for article in articles:
            # Get the publish timestamp
            try:
                published_at = datetime.fromisoformat(article.get("published_at", now.isoformat()))
            except:
                published_at = now
                
            # Calculate recency weight (more recent = higher weight)
            hours_ago = (now - published_at).total_seconds() / 3600
            recency_weight = np.exp(-hours_ago / 24)  # Exponential decay with 24-hour half-life
            
            # Get sentiment (using simulated value)
            sentiment = article.get("_simulated_sentiment", 0)
            
            weighted_sentiments.append((sentiment, recency_weight))
            texts.append(article.get("headline", "") + " " + article.get("summary", ""))
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_sentiments)
        if total_weight > 0:
            avg_sentiment = sum(sentiment * weight for sentiment, weight in weighted_sentiments) / total_weight
        else:
            avg_sentiment = 0
            
        return {
            "score": avg_sentiment,
            "count": len(articles),
            "texts": texts,
            "label": self._get_sentiment_label(avg_sentiment)
        }
    
    def _analyze_onchain_sentiment(self, metrics: Dict) -> Dict:
        """
        Analyze sentiment from on-chain metrics
        
        Args:
            metrics: Dictionary of on-chain metrics
            
        Returns:
            On-chain sentiment analysis results
        """
        if not metrics:
            return {"score": 0, "factors": {}}
            
        # Convert on-chain metrics to sentiment signals
        
        # 1. Holder growth signal (new holders as % of total)
        holder_growth = metrics.get("new_holders_24h", 0) / max(1, metrics.get("unique_holders", 1))
        holder_growth_signal = np.clip(holder_growth * 100 - 1, -1, 1)  # -1 to +1 range
        
        # 2. Activity signal (daily active / total holders)
        activity_ratio = metrics.get("daily_active_addresses", 0) / max(1, metrics.get("unique_holders", 1))
        activity_signal = np.clip(activity_ratio * 5 - 0.5, -1, 1)  # -1 to +1 range
        
        # 3. Buy/sell ratio signal
        bs_ratio = metrics.get("buy_sell_ratio", 1.0)
        bs_ratio_signal = np.clip((bs_ratio - 1) * 2, -1, 1)  # -1 to +1 range
        
        # 4. Whale concentration signal (negative signal if too concentrated)
        concentration = metrics.get("holder_concentration", 50)
        concentration_signal = np.clip(1 - (concentration / 50), -1, 1)  # -1 to +1 range
        
        # Combine signals with weights
        sentiment_score = (
            holder_growth_signal * 0.3 +
            activity_signal * 0.25 +
            bs_ratio_signal * 0.25 +
            concentration_signal * 0.2
        )
        
        return {
            "score": sentiment_score,
            "factors": {
                "holder_growth": holder_growth_signal,
                "activity": activity_signal,
                "buy_sell_ratio": bs_ratio_signal,
                "concentration": concentration_signal
            },
            "label": self._get_sentiment_label(sentiment_score)
        }
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """
        Convert sentiment score to human-readable label
        
        Args:
            sentiment_score: Sentiment score (-1 to +1)
            
        Returns:
            Sentiment label
        """
        if sentiment_score < self.sentiment_thresholds["very_negative"]:
            return "Very Negative"
        elif sentiment_score < self.sentiment_thresholds["negative"]:
            return "Negative"
        elif sentiment_score < self.sentiment_thresholds["neutral"]:
            return "Slightly Negative"
        elif sentiment_score < self.sentiment_thresholds["positive"]:
            return "Slightly Positive"
        elif sentiment_score < self.sentiment_thresholds["very_positive"]:
            return "Positive"
        else:
            return "Very Positive"
    
    def _extract_key_topics(self, texts: List[str], num_topics: int = 3, num_words: int = 5) -> List[Dict]:
        """
        Extract key topics from text data using LDA
        
        Args:
            texts: List of text strings
            num_topics: Number of topics to extract
            num_words: Number of words per topic
            
        Returns:
            List of topics with key words
        """
        if not texts or len(texts) < 5:
            return []
            
        # Preprocess texts
        processed_texts = []
        for text in texts:
            if not text:
                continue
                
            # Simple preprocessing (lowercase, remove special chars)
            processed = re.sub(r'[^\w\s]', '', text.lower())
            processed_texts.append(processed)
            
        if not processed_texts:
            return []
            
        # Create TF-IDF representation
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
            X = vectorizer.fit_transform(processed_texts)
            
            # Check if we have enough data
            if X.shape[0] < 3 or X.shape[1] < 5:
                return []
                
            # Run LDA
            lda = LatentDirichletAllocation(
                n_components=min(num_topics, len(processed_texts) // 2),
                random_state=42,
                max_iter=10
            )
            lda.fit(X)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    "id": topic_idx,
                    "words": top_words,
                    "weight": float(topic.sum() / lda.components_.sum())
                })
                
            return topics
        except Exception as e:
            # Fallback if topic modeling fails
            return []
    
    def _generate_insights(self, overall_sentiment: float, source_sentiments: Dict, topics: List[Dict]) -> List[str]:
        """
        Generate personalized insights based on sentiment analysis
        
        Args:
            overall_sentiment: Overall sentiment score
            source_sentiments: Sentiment scores by source
            topics: Extracted topics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overall sentiment insight
        sentiment_label = self._get_sentiment_label(overall_sentiment)
        insights.append(f"Overall market sentiment is {sentiment_label.lower()} with a score of {overall_sentiment:.2f}")
        
        # Source comparison insights
        sources = list(source_sentiments.keys())
        source_scores = [(source, data["score"]) for source, data in source_sentiments.items() 
                         if isinstance(data, dict) and "score" in data]
        
        # Sort by sentiment score
        source_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(source_scores) >= 2:
            most_positive = source_scores[0]
            most_negative = source_scores[-1]
            
            if most_positive[1] - most_negative[1] > 0.3:
                insights.append(
                    f"Sentiment divergence detected: {most_positive[0].capitalize()} data shows "
                    f"{self._get_sentiment_label(most_positive[1]).lower()} sentiment, while "
                    f"{most_negative[0].capitalize()} data shows {self._get_sentiment_label(most_negative[1]).lower()} sentiment"
                )
                
        # Topic insights
        if topics:
            topic_list = ", ".join([", ".join(topic["words"][:3]) for topic in topics[:2]])
            insights.append(f"Key topics in discussions: {topic_list}")
        
        # On-chain insights
        onchain = source_sentiments.get("onchain", {})
        if isinstance(onchain, dict) and "factors" in onchain:
            factors = onchain["factors"]
            
            # Add insights for significant on-chain signals
            if "buy_sell_ratio" in factors and abs(factors["buy_sell_ratio"]) > 0.5:
                direction = "buying" if factors["buy_sell_ratio"] > 0 else "selling"
                intensity = "strong" if abs(factors["buy_sell_ratio"]) > 0.8 else "moderate"
                insights.append(f"On-chain data shows {intensity} {direction} pressure")
                
            if "concentration" in factors and factors["concentration"] < -0.5:
                insights.append("High whale concentration detected, indicating potential volatility risk")
                
            if "holder_growth" in factors and factors["holder_growth"] > 0.8:
                insights.append("Significant recent growth in new token holders, suggesting rising interest")
                
        return insights
    
    def _generate_recommendations(self, sentiment_score: float, topics: List[Dict], onchain_data: Dict) -> List[str]:
        """
        Generate personalized trading recommendations
        
        Args:
            sentiment_score: Overall sentiment score
            topics: Extracted topics
            onchain_data: On-chain metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Base recommendations on sentiment score
        if sentiment_score > 0.7:
            recommendations.append("Strong positive sentiment suggests favorable short-term price potential")
        elif sentiment_score > 0.4:
            recommendations.append("Positive sentiment indicates potential buying opportunity with reasonable stop-loss")
        elif sentiment_score > 0.1:
            recommendations.append("Slightly positive sentiment suggests cautious optimism")
        elif sentiment_score > -0.1:
            recommendations.append("Neutral sentiment suggests holding positions and monitoring for clear signals")
        elif sentiment_score > -0.4:
            recommendations.append("Slightly negative sentiment suggests caution with new positions")
        elif sentiment_score > -0.7:
            recommendations.append("Negative sentiment indicates potential near-term downside, consider reducing exposure")
        else:
            recommendations.append("Strong negative sentiment suggests significant risk, consider hedging or exiting positions")
            
        # Additional recommendations based on on-chain data
        buy_sell_ratio = onchain_data.get("buy_sell_ratio", 1.0)
        if buy_sell_ratio > 1.3 and sentiment_score > 0:
            recommendations.append("Strong buying pressure aligned with positive sentiment suggests potential upward movement")
        elif buy_sell_ratio < 0.7 and sentiment_score < 0:
            recommendations.append("Strong selling pressure aligned with negative sentiment suggests further downside risk")
            
        # Holder concentration risk
        concentration = onchain_data.get("holder_concentration", 50)
        if concentration > 80:
            recommendations.append("Extreme holder concentration presents significant volatility risk")
            
        # Growth indicators
        holder_growth = onchain_data.get("new_holders_24h", 0) / max(1, onchain_data.get("unique_holders", 1))
        if holder_growth > 0.05:  # More than 5% new holders in 24h
            recommendations.append("Rapid growth in new holders indicates surging interest, monitor for momentum shift")
            
        return recommendations 