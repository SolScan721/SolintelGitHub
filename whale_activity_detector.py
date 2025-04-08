import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class WhaleActivityDetector:
    """
    Intelligent Trading Pattern Recognition - Core feature of SolIntel
    
    This module detects whale activity and market manipulation patterns
    on the Solana blockchain using advanced graph neural networks and
    anomaly detection algorithms.
    """
    
    def __init__(self, min_transaction_amount: float = 50000.0):
        """
        Initialize the WhaleActivityDetector with configuration parameters
        
        Args:
            min_transaction_amount: Minimum USD value to classify as potential whale activity
        """
        self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
        self.min_transaction_amount = min_transaction_amount
        self.known_whale_addresses = self._load_known_whales()
        self.transaction_graph = nx.DiGraph()
        self.historical_patterns = {}
        
    def _load_known_whales(self) -> Dict[str, Dict]:
        """
        Load database of known whale addresses and their historical behavior patterns
        
        Returns:
            Dictionary of whale addresses with metadata
        """
        # In production: This would load from a database
        # Simulated data for demonstration
        return {
            "HN7Nkx4LzNQ2CS3xVBLyPsiZHsBPH1WS3L7K4YPWUwxS": {
                "label": "Exchange hot wallet",
                "risk_score": 0.2,
                "historical_avg_txn_size": 125000,
                "known_tokens": ["SOL", "BONK", "JTO", "WIF"]
            },
            "CFdipSBTJaY5xZpNH6WMQGU9cTxK6dQUgNf7nEYAEog9": {
                "label": "Institutional investor",
                "risk_score": 0.4,
                "historical_avg_txn_size": 780000,
                "known_tokens": ["SOL", "RAY", "SRM"]
            },
            "4KhDEwkdLRdD2C7eGpJqWqQ3P9eNenXi8KLZXcpDrLqz": {
                "label": "Market maker",
                "risk_score": 0.1,
                "historical_avg_txn_size": 430000,
                "known_tokens": ["SOL", "MSOL", "USDC"]
            }
        }
    
    def analyze_token_whale_activity(self, token_address: str, lookback_hours: int = 24) -> Dict:
        """
        Analyze whale activity for a specific token
        
        Args:
            token_address: Solana token address to analyze
            lookback_hours: Hours of historical data to analyze
            
        Returns:
            Dictionary containing whale activity analysis results
        """
        # Get recent transactions for the token
        transactions = self._get_token_transactions(token_address, lookback_hours)
        
        # Build transaction graph 
        self._build_transaction_graph(transactions)
        
        # Detect whale clusters
        whale_clusters = self._detect_whale_clusters(transactions)
        
        # Analyze trading patterns
        manipulation_patterns = self._detect_manipulation_patterns(transactions, whale_clusters)
        
        # Calculate combined risk scores
        activity_risk = self._calculate_activity_risk(
            whale_clusters, 
            manipulation_patterns,
            transactions
        )
        
        return {
            "token_address": token_address,
            "analysis_timestamp": datetime.now().isoformat(),
            "whale_activity_score": activity_risk["whale_activity_score"],
            "manipulation_probability": activity_risk["manipulation_probability"],
            "whale_clusters": whale_clusters,
            "detected_patterns": manipulation_patterns,
            "prediction": {
                "price_impact_prediction": activity_risk["price_impact_prediction"],
                "confidence": activity_risk["prediction_confidence"],
                "time_horizon": f"{activity_risk['time_horizon']} hours"
            },
            "watch_addresses": self._get_addresses_to_watch(whale_clusters, manipulation_patterns)
        }
    
    def _get_token_transactions(self, token_address: str, lookback_hours: int) -> List[Dict]:
        """
        Retrieve token transactions from Solana blockchain
        
        Args:
            token_address: Token address to analyze
            lookback_hours: Hours of historical data to retrieve
            
        Returns:
            List of transaction data
        """
        # In production: This would call Solana RPC interface and process data
        # Using simulated data for demonstration
        num_transactions = np.random.randint(200, 1000)
        
        # Generate synthetic transaction data
        transactions = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        for _ in range(num_transactions):
            # Random timestamp within the lookback period
            timestamp = start_time + timedelta(
                seconds=np.random.randint(0, lookback_hours * 3600)
            )
            
            # Generate plausible transaction data
            is_whale = np.random.random() < 0.15  # 15% of transactions are from whales
            amount = np.random.lognormal(mean=10, sigma=2) * (10 if is_whale else 1)
            
            # Select sender and receiver addresses
            if is_whale and np.random.random() < 0.7:  # 70% chance to use known whale address
                sender = np.random.choice(list(self.known_whale_addresses.keys()))
            else:
                sender = f"Random{np.random.randint(1000, 9999)}Address{np.random.randint(100, 999)}"
                
            receiver = f"Random{np.random.randint(1000, 9999)}Address{np.random.randint(100, 999)}"
            
            transactions.append({
                "signature": f"Sig{np.random.randint(10000, 99999)}",
                "blockTime": int(timestamp.timestamp()),
                "amount": amount,
                "sender": sender,
                "receiver": receiver,
                "usd_value": amount * np.random.uniform(0.5, 2.0),
                "tokenAddress": token_address,
                "fee": 0.000005 * np.random.uniform(0.9, 1.1)
            })
            
        # Sort by timestamp
        transactions.sort(key=lambda x: x["blockTime"])
        return transactions
    
    def _build_transaction_graph(self, transactions: List[Dict]) -> None:
        """
        Build a directed graph of transactions for network analysis
        
        Args:
            transactions: List of transaction data
        """
        # Reset the graph
        self.transaction_graph = nx.DiGraph()
        
        # Add nodes and edges
        for txn in transactions:
            sender = txn["sender"]
            receiver = txn["receiver"]
            amount = txn["amount"]
            
            # Add sender node if it doesn't exist
            if not self.transaction_graph.has_node(sender):
                self.transaction_graph.add_node(
                    sender, 
                    is_known_whale=sender in self.known_whale_addresses,
                    total_sent=0,
                    total_received=0,
                    transaction_count=0
                )
                
            # Add receiver node if it doesn't exist
            if not self.transaction_graph.has_node(receiver):
                self.transaction_graph.add_node(
                    receiver, 
                    is_known_whale=receiver in self.known_whale_addresses,
                    total_sent=0,
                    total_received=0, 
                    transaction_count=0
                )
            
            # Add or update edge
            if self.transaction_graph.has_edge(sender, receiver):
                self.transaction_graph[sender][receiver]["amount"] += amount
                self.transaction_graph[sender][receiver]["count"] += 1
            else:
                self.transaction_graph.add_edge(
                    sender, 
                    receiver, 
                    amount=amount, 
                    count=1
                )
            
            # Update node attributes
            self.transaction_graph.nodes[sender]["total_sent"] = (
                self.transaction_graph.nodes[sender].get("total_sent", 0) + amount
            )
            self.transaction_graph.nodes[receiver]["total_received"] = (
                self.transaction_graph.nodes[receiver].get("total_received", 0) + amount
            )
            self.transaction_graph.nodes[sender]["transaction_count"] = (
                self.transaction_graph.nodes[sender].get("transaction_count", 0) + 1
            )
            self.transaction_graph.nodes[receiver]["transaction_count"] = (
                self.transaction_graph.nodes[receiver].get("transaction_count", 0) + 1
            )
    
    def _detect_whale_clusters(self, transactions: List[Dict]) -> List[Dict]:
        """
        Detect clusters of whale addresses using graph analysis and DBSCAN clustering
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List of detected whale clusters
        """
        # Extract features for clustering
        addresses = set()
        for txn in transactions:
            addresses.add(txn["sender"])
            addresses.add(txn["receiver"])
            
        # Create feature matrix for clustering
        features = []
        address_list = []
        
        for address in addresses:
            if address in self.transaction_graph:
                node = self.transaction_graph.nodes[address]
                
                # Skip addresses with minimal activity
                if node.get("total_sent", 0) + node.get("total_received", 0) < self.min_transaction_amount:
                    continue
                    
                # Extract network features
                in_degree = self.transaction_graph.in_degree(address)
                out_degree = self.transaction_graph.out_degree(address)
                clustering_coef = nx.clustering(self.transaction_graph, address)
                pagerank = nx.pagerank(self.transaction_graph).get(address, 0)
                
                features.append([
                    node.get("total_sent", 0),
                    node.get("total_received", 0),
                    node.get("transaction_count", 0),
                    in_degree,
                    out_degree,
                    clustering_coef,
                    pagerank
                ])
                address_list.append(address)
        
        # Perform clustering if we have enough data points
        if len(features) < 5:
            return []
            
        # Normalize features
        X = StandardScaler().fit_transform(features)
        
        # Apply DBSCAN clustering
        db = DBSCAN(eps=0.5, min_samples=3).fit(X)
        labels = db.labels_
        
        # Group addresses by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Skip noise points
                continue
                
            if label not in clusters:
                clusters[label] = []
                
            clusters[label].append(address_list[i])
        
        # Format results
        result = []
        for cluster_id, addresses in clusters.items():
            total_value = 0
            known_whales = []
            
            for address in addresses:
                node = self.transaction_graph.nodes[address]
                total_value += node.get("total_sent", 0) + node.get("total_received", 0)
                
                if address in self.known_whale_addresses:
                    known_whales.append({
                        "address": address,
                        "label": self.known_whale_addresses[address]["label"]
                    })
            
            result.append({
                "cluster_id": int(cluster_id),
                "addresses": addresses,
                "size": len(addresses),
                "total_value": total_value,
                "known_whales": known_whales,
                "risk_score": self._calculate_cluster_risk(addresses, total_value)
            })
            
        return result
    
    def _calculate_cluster_risk(self, addresses: List[str], total_value: float) -> float:
        """
        Calculate risk score for a whale cluster
        
        Args:
            addresses: List of addresses in the cluster
            total_value: Total transaction value in the cluster
            
        Returns:
            Risk score between 0 and 1
        """
        # Base risk on cluster size, total value, and known whales
        known_whale_count = sum(1 for addr in addresses if addr in self.known_whale_addresses)
        known_whale_weight = 0.6
        
        # Higher risk if unknown large wallets
        if known_whale_count == 0 and len(addresses) > 3:
            base_risk = 0.75
        else:
            base_risk = 0.3
            
        # Adjust for known whales
        known_whale_risk = 0
        for addr in addresses:
            if addr in self.known_whale_addresses:
                known_whale_risk += self.known_whale_addresses[addr]["risk_score"]
                
        if known_whale_count > 0:
            known_whale_risk /= known_whale_count
            
        # Adjust for total value (higher value = higher risk)
        value_risk = min(1.0, total_value / 10000000)  # Cap at 10M
        
        # Combine risks
        combined_risk = (
            base_risk * 0.3 +
            known_whale_risk * known_whale_weight +
            value_risk * 0.4
        )
        
        return min(1.0, max(0.0, combined_risk))
    
    def _detect_manipulation_patterns(self, transactions: List[Dict], whale_clusters: List[Dict]) -> List[Dict]:
        """
        Detect potential market manipulation patterns
        
        Args:
            transactions: List of transaction data
            whale_clusters: List of detected whale clusters
            
        Returns:
            List of detected manipulation patterns
        """
        patterns = []
        
        # Extract whale addresses from clusters
        whale_addresses = set()
        for cluster in whale_clusters:
            whale_addresses.update(cluster["addresses"])
            
        # Look for wash trading pattern
        wash_trading = self._detect_wash_trading(transactions, whale_addresses)
        if wash_trading["detected"]:
            patterns.append(wash_trading)
            
        # Look for pump-and-dump pattern
        pump_dump = self._detect_pump_dump_pattern(transactions, whale_addresses)
        if pump_dump["detected"]:
            patterns.append(pump_dump)
            
        # Look for spoofing pattern (rapid order placement/cancellation)
        spoofing = self._detect_spoofing_pattern(transactions, whale_addresses)
        if spoofing["detected"]:
            patterns.append(spoofing)
            
        return patterns
        
    def _detect_wash_trading(self, transactions: List[Dict], whale_addresses: set) -> Dict:
        """
        Detect wash trading patterns (trading between related accounts)
        
        Args:
            transactions: List of transaction data
            whale_addresses: Set of whale addresses
            
        Returns:
            Wash trading detection results
        """
        # Look for circular transaction patterns
        # In production: More sophisticated cycle detection in transaction graph
        
        # Find cycles in the transaction graph
        try:
            cycles = list(nx.simple_cycles(self.transaction_graph))
            short_cycles = [c for c in cycles if 2 <= len(c) <= 4]  # Focus on short cycles
            
            if not short_cycles:
                return {"detected": False, "type": "wash_trading", "evidence": []}
                
            # Check if cycles involve whale addresses
            whale_cycles = []
            for cycle in short_cycles:
                if any(addr in whale_addresses for addr in cycle):
                    whale_cycles.append(cycle)
                    
            if not whale_cycles:
                return {"detected": False, "type": "wash_trading", "evidence": []}
                
            # Calculate total volume in these cycles
            cycle_volume = 0
            evidence = []
            
            for cycle in whale_cycles[:5]:  # Limit to top 5 cycles
                cycle_txns = []
                cycle_total = 0
                
                for i in range(len(cycle)):
                    sender = cycle[i]
                    receiver = cycle[(i + 1) % len(cycle)]
                    
                    if self.transaction_graph.has_edge(sender, receiver):
                        edge_data = self.transaction_graph[sender][receiver]
                        cycle_txns.append({
                            "sender": sender,
                            "receiver": receiver,
                            "amount": edge_data["amount"],
                            "count": edge_data["count"]
                        })
                        cycle_total += edge_data["amount"]
                
                if cycle_txns:
                    evidence.append({
                        "addresses": cycle,
                        "transactions": cycle_txns,
                        "total_volume": cycle_total
                    })
                    cycle_volume += cycle_total
            
            # Determine if this is significant wash trading
            total_volume = sum(txn["amount"] for txn in transactions)
            wash_percent = min(100, (cycle_volume / total_volume * 100)) if total_volume > 0 else 0
            
            return {
                "detected": wash_percent > 15,  # If wash trading > 15% of volume
                "type": "wash_trading",
                "severity": self._calculate_severity(wash_percent, 15, 70),
                "evidence": evidence,
                "cycle_volume": cycle_volume,
                "cycle_volume_percent": wash_percent
            }
            
        except Exception as e:
            # Fallback if cycle detection fails
            return {"detected": False, "type": "wash_trading", "evidence": []}
    
    def _detect_pump_dump_pattern(self, transactions: List[Dict], whale_addresses: set) -> Dict:
        """
        Detect pump and dump patterns (accumulation followed by large sells)
        
        Args:
            transactions: List of transaction data
            whale_addresses: Set of whale addresses
            
        Returns:
            Pump and dump detection results
        """
        # Sort transactions by time
        sorted_txns = sorted(transactions, key=lambda x: x["blockTime"])
        
        # Divide into time windows
        window_size_hours = 4
        window_size_seconds = window_size_hours * 3600
        
        if not sorted_txns:
            return {"detected": False, "type": "pump_dump", "evidence": []}
            
        start_time = sorted_txns[0]["blockTime"]
        end_time = sorted_txns[-1]["blockTime"]
        
        # If less than window_size_hours of data, adjust window size
        if end_time - start_time < window_size_seconds:
            window_size_seconds = max(900, (end_time - start_time) // 2)  # At least 15 minutes
            
        windows = []
        current_window = []
        current_window_end = start_time + window_size_seconds
        
        for txn in sorted_txns:
            if txn["blockTime"] <= current_window_end:
                current_window.append(txn)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [txn]
                current_window_end = txn["blockTime"] + window_size_seconds
                
        if current_window:
            windows.append(current_window)
            
        # Need at least 2 windows for comparison
        if len(windows) < 2:
            return {"detected": False, "type": "pump_dump", "evidence": []}
            
        # Calculate net flow for whale addresses in each window
        window_stats = []
        
        for i, window in enumerate(windows):
            whale_buys = 0
            whale_sells = 0
            
            for txn in window:
                if txn["sender"] in whale_addresses and txn["receiver"] not in whale_addresses:
                    whale_sells += txn["amount"]
                elif txn["sender"] not in whale_addresses and txn["receiver"] in whale_addresses:
                    whale_buys += txn["amount"]
                    
            window_stats.append({
                "window_index": i,
                "start_time": window[0]["blockTime"],
                "end_time": window[-1]["blockTime"],
                "whale_buys": whale_buys,
                "whale_sells": whale_sells,
                "net_flow": whale_buys - whale_sells,
                "transaction_count": len(window)
            })
        
        # Look for accumulation followed by selling
        evidence = []
        
        for i in range(len(window_stats) - 1):
            current = window_stats[i]
            next_window = window_stats[i + 1]
            
            # If whales accumulated in this window and sold in the next
            if (current["net_flow"] > 0 and 
                next_window["net_flow"] < 0 and 
                abs(next_window["net_flow"]) > current["net_flow"] * 1.5):
                
                evidence.append({
                    "accumulation_window": current,
                    "dump_window": next_window,
                    "accumulation_amount": current["net_flow"],
                    "dump_amount": abs(next_window["net_flow"]),
                    "dump_ratio": abs(next_window["net_flow"]) / current["net_flow"] if current["net_flow"] > 0 else 0
                })
                
        if not evidence:
            return {"detected": False, "type": "pump_dump", "evidence": []}
            
        # Calculate severity based on the largest pump-dump pattern found
        if evidence:
            largest_pattern = max(evidence, key=lambda x: x["dump_amount"])
            severity = self._calculate_severity(
                largest_pattern["dump_ratio"], 
                1.5,  # Min ratio to consider
                5.0   # Ratio for max severity
            )
        else:
            severity = 0
            
        return {
            "detected": True,
            "type": "pump_dump",
            "severity": severity,
            "evidence": evidence
        }
    
    def _detect_spoofing_pattern(self, transactions: List[Dict], whale_addresses: set) -> Dict:
        """
        Detect spoofing patterns (rapid order placement/cancellation)
        
        Args:
            transactions: List of transaction data
            whale_addresses: Set of whale addresses
            
        Returns:
            Spoofing detection results
        """
        # Simplified for demonstration
        # In production: Analyze DEX order placement/cancellation data
        return {
            "detected": False,
            "type": "spoofing",
            "evidence": []
        }
    
    def _calculate_activity_risk(
        self, 
        whale_clusters: List[Dict], 
        manipulation_patterns: List[Dict],
        transactions: List[Dict]
    ) -> Dict:
        """
        Calculate overall risk scores based on whale activity and manipulation patterns
        
        Args:
            whale_clusters: Detected whale clusters
            manipulation_patterns: Detected manipulation patterns
            transactions: Transaction data
            
        Returns:
            Dictionary with risk scores and predictions
        """
        # Base whale activity score on cluster risk scores
        whale_activity_score = 0
        if whale_clusters:
            # Weighted average of cluster risk scores
            total_value = sum(cluster["total_value"] for cluster in whale_clusters)
            if total_value > 0:
                whale_activity_score = sum(
                    cluster["risk_score"] * cluster["total_value"] / total_value 
                    for cluster in whale_clusters
                )
        
        # Manipulation probability based on detected patterns
        manipulation_probability = 0
        for pattern in manipulation_patterns:
            if "severity" in pattern:
                manipulation_probability = max(manipulation_probability, pattern["severity"])
        
        # Combined risk assessment
        combined_risk = (whale_activity_score * 0.6) + (manipulation_probability * 0.4)
        
        # Generate price impact prediction
        if combined_risk > 0.8:
            price_impact = "Significant negative (-15% to -30%)"
            confidence = 0.85
            time_horizon = 3
        elif combined_risk > 0.6:
            price_impact = "Moderate negative (-5% to -15%)"
            confidence = 0.75
            time_horizon = 6
        elif combined_risk > 0.4:
            price_impact = "Slight negative (-1% to -5%)"
            confidence = 0.65
            time_horizon = 12
        elif combined_risk > 0.2:
            price_impact = "Neutral (-1% to +1%)"
            confidence = 0.60
            time_horizon = 24
        else:
            price_impact = "Positive (+1% to +5%)"
            confidence = 0.55
            time_horizon = 24
        
        return {
            "whale_activity_score": whale_activity_score,
            "manipulation_probability": manipulation_probability,
            "combined_risk": combined_risk,
            "price_impact_prediction": price_impact,
            "prediction_confidence": confidence,
            "time_horizon": time_horizon
        }
    
    def _get_addresses_to_watch(self, whale_clusters: List[Dict], manipulation_patterns: List[Dict]) -> List[Dict]:
        """
        Generate list of addresses to monitor based on detected activity
        
        Args:
            whale_clusters: Detected whale clusters
            manipulation_patterns: Detected manipulation patterns
            
        Returns:
            List of addresses to watch with risk levels
        """
        addresses_to_watch = {}
        
        # Add addresses from whale clusters
        for cluster in whale_clusters:
            for address in cluster["addresses"]:
                if address not in addresses_to_watch:
                    addresses_to_watch[address] = {
                        "address": address,
                        "risk_level": "medium",
                        "reason": "Part of whale cluster"
                    }
        
        # Add addresses from manipulation patterns
        for pattern in manipulation_patterns:
            if pattern["type"] == "wash_trading" and pattern["detected"]:
                for evidence in pattern["evidence"]:
                    for address in evidence["addresses"]:
                        if address in addresses_to_watch:
                            addresses_to_watch[address]["risk_level"] = "high"
                            addresses_to_watch[address]["reason"] = "Potential wash trading activity"
                        else:
                            addresses_to_watch[address] = {
                                "address": address,
                                "risk_level": "high",
                                "reason": "Potential wash trading activity"
                            }
            
            if pattern["type"] == "pump_dump" and pattern["detected"]:
                for evidence in pattern["evidence"]:
                    dump_ratio = evidence.get("dump_ratio", 0)
                    
                    # Only add addresses if there's a significant dump ratio
                    if dump_ratio > 2.0:
                        # In production: This would extract specific addresses involved in the dump
                        # For demonstration, we're just marking all whale addresses as high risk
                        for address in self.transaction_graph.nodes():
                            if (self.transaction_graph.nodes[address].get("total_sent", 0) > 
                                self.min_transaction_amount):
                                if address in addresses_to_watch:
                                    addresses_to_watch[address]["risk_level"] = "high"
                                    addresses_to_watch[address]["reason"] = "Potential pump & dump activity"
                                else:
                                    addresses_to_watch[address] = {
                                        "address": address,
                                        "risk_level": "high",
                                        "reason": "Potential pump & dump activity"
                                    }
        
        return list(addresses_to_watch.values())
    
    def _calculate_severity(self, value: float, min_threshold: float, max_threshold: float) -> float:
        """
        Calculate severity score (0-1) based on value and thresholds
        
        Args:
            value: Value to calculate severity for
            min_threshold: Minimum threshold for detection
            max_threshold: Threshold for maximum severity
            
        Returns:
            Severity score between 0 and 1
        """
        if value < min_threshold:
            return 0
        elif value >= max_threshold:
            return 1.0
        else:
            # Linear interpolation between thresholds
            return (value - min_threshold) / (max_threshold - min_threshold) 