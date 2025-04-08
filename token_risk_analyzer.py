import os
import requests
import numpy as np
from datetime import datetime
import time

class TokenRiskAnalyzer:
    """Token Risk Analysis Module - Core feature of SolIntel"""
    
    def __init__(self):
        self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
    
    def analyze_token(self, token_address):
        """
        Analyze token risk metrics
        
        Args:
            token_address (str): Solana token address
            
        Returns:
            dict: Dictionary containing risk analysis results
        """
        # Get token basic information
        token_info = self._get_token_info(token_address)
        
        # Analyze liquidity risk
        liquidity_risk = self._analyze_liquidity(token_address)
        
        # Analyze code security risk
        code_risk = self._analyze_contract_security(token_address)
        
        # Analyze transaction pattern risk
        transaction_risk = self._analyze_transaction_patterns(token_address)
        
        # Calculate overall risk score (1-100, 100 being safest)
        overall_risk_score = self._calculate_overall_risk(
            liquidity_risk, 
            code_risk, 
            transaction_risk
        )
        
        return {
            "token_info": token_info,
            "risk_analysis": {
                "overall_risk_score": overall_risk_score,
                "risk_level": self._get_risk_level(overall_risk_score),
                "liquidity_risk": liquidity_risk,
                "code_risk": code_risk,
                "transaction_risk": transaction_risk
            },
            "recommendations": self._generate_recommendations(overall_risk_score),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _get_token_info(self, token_address):
        """Get token basic information"""
        # In real implementation, this would call Solana RPC interface
        # This is just a simulation
        return {
            "address": token_address,
            "name": f"Token {token_address[:6]}",
            "symbol": f"TKN{token_address[:3]}",
            "total_supply": 1000000000,
            "decimals": 9,
            "creation_time": int(time.time()) - 86400 * np.random.randint(1, 100)
        }
    
    def _analyze_liquidity(self, token_address):
        """Analyze token liquidity risk"""
        # Simulate liquidity analysis results
        # In real implementation, would analyze DEX liquidity pool data
        liquidity_factors = {
            "pool_count": np.random.randint(1, 10),
            "largest_pool_size_usd": np.random.randint(10000, 1000000),
            "liquidity_concentration": np.random.random() * 100,
            "price_impact_1000usd": np.random.random() * 10
        }
        
        # Liquidity risk score (0-100, 0 being highest risk)
        score = min(100, max(0, 
            30 * min(liquidity_factors["pool_count"] / 5, 1) +
            40 * min(liquidity_factors["largest_pool_size_usd"] / 500000, 1) +
            30 * (1 - liquidity_factors["price_impact_1000usd"] / 10)
        ))
        
        return {
            "score": score,
            "factors": liquidity_factors
        }
    
    def _analyze_contract_security(self, token_address):
        """Analyze token contract security risk"""
        # Simulate code security analysis results
        # In real implementation, would analyze contract code security features
        security_factors = {
            "has_mint_function": np.random.choice([True, False], p=[0.3, 0.7]),
            "has_blacklist_function": np.random.choice([True, False], p=[0.4, 0.6]),
            "has_ownership_transfer": np.random.choice([True, False], p=[0.6, 0.4]),
            "is_proxy_contract": np.random.choice([True, False], p=[0.2, 0.8]),
            "similar_to_known_scams": np.random.choice([True, False], p=[0.1, 0.9])
        }
        
        # Security risk score (0-100, 0 being highest risk)
        base_score = 100
        if security_factors["has_mint_function"]:
            base_score -= 30
        if security_factors["is_proxy_contract"]:
            base_score -= 20
        if security_factors["similar_to_known_scams"]:
            base_score -= 50
        if not security_factors["has_blacklist_function"]:
            base_score -= 10
            
        score = min(100, max(0, base_score))
        
        return {
            "score": score,
            "factors": security_factors
        }
    
    def _analyze_transaction_patterns(self, token_address):
        """Analyze token transaction pattern risk"""
        # Simulate transaction pattern analysis results
        # In real implementation, would analyze on-chain transaction patterns
        pattern_factors = {
            "wash_trading_probability": np.random.random() * 0.5,
            "price_manipulation_signs": np.random.choice([True, False], p=[0.2, 0.8]),
            "whale_concentration": np.random.random() * 100,
            "trading_volume_consistency": np.random.random()
        }
        
        # Transaction pattern risk score (0-100, 0 being highest risk)
        score = min(100, max(0, 
            30 * (1 - pattern_factors["wash_trading_probability"] * 2) +
            40 * (not pattern_factors["price_manipulation_signs"]) +
            30 * (1 - min(pattern_factors["whale_concentration"] / 80, 1))
        ))
        
        return {
            "score": score,
            "factors": pattern_factors
        }
    
    def _calculate_overall_risk(self, liquidity_risk, code_risk, transaction_risk):
        """Calculate overall risk score"""
        # Weights: Liquidity 40%, Code Security 35%, Transaction Patterns 25%
        return (
            0.40 * liquidity_risk["score"] +
            0.35 * code_risk["score"] +
            0.25 * transaction_risk["score"]
        )
    
    def _get_risk_level(self, risk_score):
        """Determine risk level based on risk score"""
        if risk_score >= 80:
            return "Low Risk"
        elif risk_score >= 60:
            return "Medium Risk"
        elif risk_score >= 40:
            return "High Risk"
        else:
            return "Extreme Risk"
    
    def _generate_recommendations(self, risk_score):
        """Generate recommendations based on risk score"""
        if risk_score >= 80:
            return [
                "This token shows low risk, but still exercise caution",
                "Consider setting stop-loss to protect against market volatility",
                "Diversify your investments to avoid concentration risk"
            ]
        elif risk_score >= 60:
            return [
                "This token carries moderate risk, consider small positions",
                "Monitor project development and community activity closely",
                "Implement strict stop-loss strategies"
            ]
        elif risk_score >= 40:
            return [
                "This token presents high risk, exercise extreme caution",
                "Only invest funds you can afford to lose completely",
                "Monitor team activity and contract changes closely"
            ]
        else:
            return [
                "This token poses extreme risk, investment not recommended",
                "Multiple indicators suggest potential scam activity",
                "Consider looking for safer investment opportunities"
            ] 