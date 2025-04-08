from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from token_risk_analyzer import TokenRiskAnalyzer
from whale_activity_detector import WhaleActivityDetector
from sentiment_analyzer import SolanaMarketSentimentAnalyzer

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize analysis modules
risk_analyzer = TokenRiskAnalyzer()
whale_detector = WhaleActivityDetector()
sentiment_analyzer = SolanaMarketSentimentAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-token', methods=['POST'])
def analyze_token():
    data = request.json
    token_address = data.get('token_address')
    token_symbol = data.get('token_symbol')
    
    if not token_address:
        return jsonify({'error': 'Token address is required'}), 400
    
    try:
        # Perform comprehensive token analysis
        risk_analysis = risk_analyzer.analyze_token(token_address)
        
        # Optional deeper analysis
        include_whale_analysis = data.get('include_whale_analysis', False)
        include_sentiment_analysis = data.get('include_sentiment_analysis', False)
        
        result = {
            "token_info": risk_analysis["token_info"],
            "risk_analysis": risk_analysis["risk_analysis"],
            "recommendations": risk_analysis["recommendations"],
            "analysis_timestamp": risk_analysis["analysis_timestamp"]
        }
        
        # Add whale activity analysis if requested
        if include_whale_analysis:
            lookback_hours = data.get('lookback_hours', 24)
            whale_analysis = whale_detector.analyze_token_whale_activity(token_address, lookback_hours)
            result["whale_analysis"] = whale_analysis
        
        # Add sentiment analysis if requested
        if include_sentiment_analysis:
            sentiment_analysis = sentiment_analyzer.analyze_token_sentiment(token_address, token_symbol)
            result["sentiment_analysis"] = sentiment_analysis
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-whale-activity', methods=['POST'])
def analyze_whale_activity():
    data = request.json
    token_address = data.get('token_address')
    lookback_hours = data.get('lookback_hours', 24)
    
    if not token_address:
        return jsonify({'error': 'Token address is required'}), 400
    
    try:
        result = whale_detector.analyze_token_whale_activity(token_address, lookback_hours)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    token_address = data.get('token_address')
    token_symbol = data.get('token_symbol')
    
    if not token_address:
        return jsonify({'error': 'Token address is required'}), 400
    
    try:
        result = sentiment_analyzer.analyze_token_sentiment(token_address, token_symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000) 