import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math


def load_and_align_data_forward_looking(price_csv_path, cot_csv_path):
    """
    Load price and COT data and align them for forward-looking analysis
    This aligns each COT report with the FOLLOWING week's price action
    """
    # Load the data
    price_data = pd.read_csv(price_csv_path)
    cot_data = pd.read_csv(cot_csv_path)

    # Convert date columns to datetime
    if 'date' in price_data.columns:
        price_data['Date'] = pd.to_datetime(price_data['date'])
    else:
        price_data['Date'] = pd.to_datetime(price_data['Date'])

    cot_data['Date'] = pd.to_datetime(cot_data['Date'])

    # Create a map of COT dates to the following week's price data
    cot_to_future_price_map = {}

    # Convert numpy datetime64 to Python datetime objects
    for cot_date in cot_data['Date'].unique():
        # Convert numpy datetime64 to Python datetime
        py_date = pd.Timestamp(cot_date).to_pydatetime()

        # Find the following week's price data (approximately 3-10 days after COT report date)
        # COT reports are published on Friday but reflect positions as of Tuesday
        # So we look at price action starting ~2-3 days after the report date
        # Start 2 days after COT date
        future_start = py_date + timedelta(days=2)
        # Look about a week ahead
        future_end = future_start + timedelta(days=8)

        # Store this mapping for later use
        cot_to_future_price_map[py_date] = (future_start, future_end)

    return price_data, cot_data, cot_to_future_price_map


def get_future_performance(pair, cot_date, price_data, cot_to_future_price_map):
    """
    Calculate the performance of a currency pair in the week following a COT report
    """
    # Convert numpy datetime64 to Python datetime if needed
    if not isinstance(cot_date, datetime):
        cot_date = pd.Timestamp(cot_date).to_pydatetime()

    future_start, future_end = cot_to_future_price_map[cot_date]

    # Filter price data for this pair and date range
    future_price_data = price_data[
        (price_data['Pair'] == pair) &
        (price_data['Date'] >= future_start) &
        (price_data['Date'] <= future_end)
    ].copy()

    # Calculate performance if we have data
    if len(future_price_data) > 0:
        future_price_data = future_price_data.sort_values('Date')
        start_price = future_price_data.iloc[0]['Close']
        end_price = future_price_data.iloc[-1]['Close']
        performance = (end_price / start_price - 1) * 100
        return performance
    else:
        return None


def analyze_currency_pair(pair, price_data, cot_data, cot_to_future_price_map):
    """
    Analyze a currency pair considering COT data
    Returns a score from -100 (very bearish) to 100 (very bullish)
    """
    # Extract base and quote currencies
    base_currency = pair.split('/')[0]
    quote_currency = pair.split('/')[1]

    # Map currency codes to COT market names
    currency_map = {
        'AUD': 'AUSTRALIAN DOLLAR',
        'GBP': 'BRITISH POUND',
        'CAD': 'CANADIAN DOLLAR',
        'EUR': 'EURO FX',
        'JPY': 'JAPANESE YEN',
        'NZD': 'NZ DOLLAR',
        'CHF': 'SWISS FRANC',
        'USD': 'US DOLLAR INDEX'
    }

    # Get the latest COT date (as Python datetime)
    latest_cot_date = pd.Timestamp(cot_data['Date'].max()).to_pydatetime()

    # Initialize sentiment scores
    base_score = 0
    quote_score = 0

    # Check if we need to perform cross-pair analysis
    if base_currency != 'USD' and quote_currency != 'USD':
        # For cross pairs, we need to analyze both currencies against USD

        # Base currency analysis
        if base_currency in currency_map:
            base_market_name = currency_map[base_currency]
            base_cot = cot_data[cot_data['Market_Name']
                                == base_market_name].copy()

            if not base_cot.empty:
                # Get the latest COT data for this currency
                latest_base_cot = base_cot[base_cot['Date'] == latest_cot_date]

                if not latest_base_cot.empty:
                    latest_base_cot = latest_base_cot.iloc[0]

                    # Calculate net position
                    comm_net = latest_base_cot['Comm_Long'] - \
                        latest_base_cot['Comm_Short']
                    retail_net = latest_base_cot['Retail_Long'] - \
                        latest_base_cot['Retail_Short']

                    # Calculate net position change from previous report
                    previous_base_cot = base_cot[base_cot['Date']
                                                 < latest_cot_date]
                    if not previous_base_cot.empty:
                        previous_base_cot = previous_base_cot.sort_values(
                            'Date', ascending=False).iloc[0]
                        prev_comm_net = previous_base_cot['Comm_Long'] - \
                            previous_base_cot['Comm_Short']
                        comm_net_change = comm_net - prev_comm_net
                        # Add momentum factor (change is often more significant than absolute level)
                        comm_net = comm_net + (comm_net_change * 2)

                    # No need for inversion in COT data as all currency futures are already
                    # in Foreign Currency/USD format
                    inversion_multiplier = 1

                    # Calculate score (commercial net position has more weight)
                    base_score = (comm_net * 0.7 + retail_net * -
                                  0.3) * inversion_multiplier

        # Quote currency analysis (similar logic)
        if quote_currency in currency_map:
            quote_market_name = currency_map[quote_currency]
            quote_cot = cot_data[cot_data['Market_Name']
                                 == quote_market_name].copy()

            if not quote_cot.empty:
                latest_quote_cot = quote_cot[quote_cot['Date']
                                             == latest_cot_date]

                if not latest_quote_cot.empty:
                    latest_quote_cot = latest_quote_cot.iloc[0]

                    comm_net = latest_quote_cot['Comm_Long'] - \
                        latest_quote_cot['Comm_Short']
                    retail_net = latest_quote_cot['Retail_Long'] - \
                        latest_quote_cot['Retail_Short']

                    inversion_multiplier = -1 if quote_currency == 'JPY' else 1

                    # For quote currency, the relationship is inverse to the pair
                    quote_score = -(comm_net * 0.7 +
                                    retail_net * -0.3) * inversion_multiplier

    else:
        # Direct USD pair analysis
        non_usd_currency = base_currency if quote_currency == 'USD' else quote_currency

        if non_usd_currency in currency_map:
            currency_market_name = currency_map[non_usd_currency]
            currency_cot = cot_data[cot_data['Market_Name']
                                    == currency_market_name].copy()

            if not currency_cot.empty:
                latest_currency_cot = currency_cot[currency_cot['Date']
                                                   == latest_cot_date]

                if not latest_currency_cot.empty:
                    latest_currency_cot = latest_currency_cot.iloc[0]

                    comm_net = latest_currency_cot['Comm_Long'] - \
                        latest_currency_cot['Comm_Short']
                    retail_net = latest_currency_cot['Retail_Long'] - \
                        latest_currency_cot['Retail_Short']

                    inversion_multiplier = -1 if non_usd_currency == 'JPY' else 1

                    # If quote is USD, normal relationship; if base is USD, inverse relationship
                    direction_multiplier = 1 if quote_currency == 'USD' else -1

                    base_score = (comm_net * 0.7 + retail_net * -0.3) * \
                        inversion_multiplier * direction_multiplier

    # Calculate price momentum (last 4 weeks)
    price_momentum = 0
    pair_price_data = price_data[price_data['Pair'] == pair].copy()

    if len(pair_price_data) >= 4:
        recent_prices = pair_price_data.sort_values(
            'Date', ascending=False).head(4)
        price_momentum = (
            recent_prices.iloc[0]['Close'] / recent_prices.iloc[-1]['Close'] - 1) * 100

    # Get historical performance pattern for this COT setup (backtest)
    # This uses historical alignment between COT positions and subsequent price movements
    historical_performance = 0
    # We use all but the most recent COT report for this analysis
    historical_cot_dates = sorted(cot_data['Date'].unique())[:-1]

    performances = []
    for hist_date in historical_cot_dates:
        # Convert to Python datetime
        py_hist_date = pd.Timestamp(hist_date).to_pydatetime()

        # Get similar COT setups from the past
        if base_currency in currency_map:
            hist_base_cot = cot_data[(cot_data['Date'] == hist_date) &
                                     (cot_data['Market_Name'] == currency_map[base_currency])]

            if not hist_base_cot.empty:
                # Calculate how similar this historical COT data is to the current setup
                # (simplified for this example)
                perf = get_future_performance(
                    pair, py_hist_date, price_data, cot_to_future_price_map)
                if perf is not None:
                    performances.append(perf)

    # Calculate average historical performance for similar setups
    if performances:
        historical_performance = sum(performances) / len(performances)

    # Combine the scores with price momentum and historical pattern
    # Use much stronger amplification for better spread but temper with historical data
    base_score = base_score * 8
    quote_score = quote_score * 8
    price_momentum = price_momentum * 4
    historical_performance = historical_performance * 6

    # Combine components for final score
    final_score = base_score + quote_score + \
        price_momentum + historical_performance

    # Normalize to -100 to 100 scale with a more balanced approach
    # Avoid excessive amplification that could lead to too many extreme readings
    scaling_factor = 60

    # Use a milder power function (square root for smoothing extreme values)
    if final_score > 0:
        amplified_score = (final_score ** 0.75)
    else:
        amplified_score = -(abs(final_score) ** 0.75)

    # Normalize with the new scaling approach
    normalized_score = min(
        max((amplified_score / scaling_factor) * 100, -100), 100)

    # Round to integer
    normalized_score = round(normalized_score)

    return normalized_score


def forecast_price_levels(pair, score, price_data, lookback_periods=12):
    """
    Forecast price levels (High, Low, Close) based on COT sentiment score
    Uses historical relationships between sentiment scores and price movements
    """
    # Filter data for this pair
    pair_data = price_data[price_data['Pair'] == pair].copy()

    if len(pair_data) < lookback_periods:
        # Not enough data for reliable forecasting
        return None

    # Get the most recent data
    pair_data = pair_data.sort_values(
        'Date', ascending=False).head(lookback_periods)

    # Get the latest close (this will be next week's open)
    latest_close = pair_data.iloc[0]['Close']
    forecast_open = latest_close

    # Calculate average weekly ranges
    avg_range = (pair_data['High'] - pair_data['Low']).mean()
    avg_high_from_open = (
        (pair_data['High'] - pair_data['Open']) / pair_data['Open'] * 100).mean()
    avg_low_from_open = (
        (pair_data['Low'] - pair_data['Open']) / pair_data['Open'] * 100).mean()

    # Calculate average weekly volatility as a percentage
    avg_volatility = avg_range / pair_data['Open'].mean() * 100

    # Adjust forecasts based on sentiment score
    # More extreme sentiment scores lead to more directional movement
    sentiment_factor = score / 30  # Normalize to approx -1 to +1 range

    # Base volatility adjustments on sentiment strength (absolute value)
    volatility_adjustment = 1.0 + \
        (abs(sentiment_factor) * 0.2)  # Up to 20% more volatile

    # For high bullish sentiment: higher highs, higher lows, higher close
    # For high bearish sentiment: lower highs, lower lows, lower close
    if sentiment_factor > 0:  # Bullish
        # Stronger bullish sentiment = higher close relative to open
        close_change_pct = avg_volatility * sentiment_factor * volatility_adjustment
        high_change_pct = avg_high_from_open * volatility_adjustment
        low_change_pct = avg_low_from_open * \
            (1 - sentiment_factor * 0.3)  # Less downside
    else:  # Bearish or neutral
        # Stronger bearish sentiment = lower close relative to open
        close_change_pct = avg_volatility * sentiment_factor * volatility_adjustment
        high_change_pct = avg_high_from_open * \
            (1 + sentiment_factor * 0.3)  # Less upside
        low_change_pct = avg_low_from_open * volatility_adjustment

    # Calculate forecasted prices
    forecast_close = forecast_open * (1 + close_change_pct/100)
    forecast_high = max(
        forecast_open * (1 + high_change_pct/100), forecast_close)
    forecast_low = min(
        forecast_open * (1 + low_change_pct/100), forecast_close)

    # Round to 5 decimal places (standard for forex)
    forecast_open = round(forecast_open, 5)
    forecast_high = round(forecast_high, 5)
    forecast_low = round(forecast_low, 5)
    forecast_close = round(forecast_close, 5)

    return {
        'Open': forecast_open,
        'High': forecast_high,
        'Low': forecast_low,
        'Close': forecast_close
    }


def generate_h4_supply_demand_zones(forecast):
    """
    Generate H4 supply and demand zones based on the weekly forecast
    """
    if forecast is None:
        return None

    open_price = forecast['Open']
    high_price = forecast['High']
    low_price = forecast['Low']
    close_price = forecast['Close']

    # Determine if forecast is bullish or bearish
    bullish = close_price > open_price

    # Calculate the total weekly range
    range_size = high_price - low_price

    # Calculate supply and demand zones
    if bullish:
        # Strong demand is near the low (0-30% of range)
        strong_demand_low = low_price
        strong_demand_high = low_price + (range_size * 0.3)

        # Moderate demand (30-50% of range)
        moderate_demand_low = low_price + (range_size * 0.3)
        moderate_demand_high = low_price + (range_size * 0.5)

        # Supply zones in upper part of range (70-100% of range)
        supply_zone_low = low_price + (range_size * 0.7)
        supply_zone_high = high_price

        # For a bullish bias, focus on buying in demand zones
        zones = [
            {
                "type": "Strong Demand",
                "zone_low": round(strong_demand_low, 5),
                "zone_high": round(strong_demand_high, 5),
                "description": "Primary buy zone with strongest bullish bias",
                "entry_strategy": "Buy on H4 bullish confirmation patterns in this zone"
            },
            {
                "type": "Moderate Demand",
                "zone_low": round(moderate_demand_low, 5),
                "zone_high": round(moderate_demand_high, 5),
                "description": "Secondary buy zone with moderate bullish bias",
                "entry_strategy": "Buy on H4 rejection of this zone with strong momentum"
            },
            {
                "type": "Supply",
                "zone_low": round(supply_zone_low, 5),
                "zone_high": round(supply_zone_high, 5),
                "description": "Potential resistance/take profit zone",
                "entry_strategy": "Take profits or reduce position size in this zone"
            }
        ]
    else:
        # Supply zones in upper part of range (70-100% of range)
        strong_supply_low = low_price + (range_size * 0.7)
        strong_supply_high = high_price

        # Moderate supply (50-70% of range)
        moderate_supply_low = low_price + (range_size * 0.5)
        moderate_supply_high = low_price + (range_size * 0.7)

        # Demand zones in lower part of range (0-30% of range)
        demand_zone_low = low_price
        demand_zone_high = low_price + (range_size * 0.3)

        # For a bearish bias, focus on selling in supply zones
        zones = [
            {
                "type": "Strong Supply",
                "zone_low": round(strong_supply_low, 5),
                "zone_high": round(strong_supply_high, 5),
                "description": "Primary sell zone with strongest bearish bias",
                "entry_strategy": "Sell on H4 bearish confirmation patterns in this zone"
            },
            {
                "type": "Moderate Supply",
                "zone_low": round(moderate_supply_low, 5),
                "zone_high": round(moderate_supply_high, 5),
                "description": "Secondary sell zone with moderate bearish bias",
                "entry_strategy": "Sell on H4 rejection of this zone with strong momentum"
            },
            {
                "type": "Demand",
                "zone_low": round(demand_zone_low, 5),
                "zone_high": round(demand_zone_high, 5),
                "description": "Potential support/take profit zone",
                "entry_strategy": "Take profits or reduce position size in this zone"
            }
        ]

    # Add additional fresh supply/demand zones based on previous week's price action
    # This would ideally use additional price data beyond the forecast

    return {
        "overall_bias": "Bullish" if bullish else "Bearish",
        "key_levels": {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "range_size": range_size
        },
        "supply_demand_zones": zones
    }


def generate_weekly_predictions_with_supply_demand(price_csv_path, cot_csv_path, pairs_list):
    """
    Generate weekly predictions with H4 supply and demand zones for a list of currency pairs
    """
    price_data, cot_data, cot_to_future_price_map = load_and_align_data_forward_looking(
        price_csv_path, cot_csv_path)

    predictions = {}

    for pair in pairs_list:
        score = analyze_currency_pair(
            pair, price_data, cot_data, cot_to_future_price_map)

        # Determine sentiment category with more granular thresholds
        if score < -30:
            sentiment = "Very Bearish"
        elif score < -15:
            sentiment = "Bearish"
        elif score < -5:
            sentiment = "Slightly Bearish"
        elif score < 5:
            sentiment = "Neutral"
        elif score < 15:
            sentiment = "Slightly Bullish"
        elif score < 30:
            sentiment = "Bullish"
        else:
            sentiment = "Very Bullish"

        # Generate price forecasts
        price_forecast = forecast_price_levels(pair, score, price_data)

        # Generate H4 supply and demand zones based on the weekly forecast
        supply_demand_zones = generate_h4_supply_demand_zones(price_forecast)

        predictions[pair] = {
            'score': round(score, 2),
            'sentiment': sentiment,
            'forecast': price_forecast,
            'supply_demand_zones': supply_demand_zones
        }

    return predictions


def get_sentiment_color(sentiment):
    """Get color based on sentiment"""
    if "Very Bearish" in sentiment:
        return "#ff0000"  # Bright red
    elif "Bearish" in sentiment:
        return "#ff5252"  # Lighter red
    elif "Slightly Bearish" in sentiment:
        return "#ff9e9e"  # Very light red
    elif "Neutral" in sentiment:
        return "#e0e0e0"  # Light gray
    elif "Slightly Bullish" in sentiment:
        return "#9effae"  # Very light green
    elif "Bullish" in sentiment:
        return "#52ff6a"  # Light green
    elif "Very Bullish" in sentiment:
        return "#00ff00"  # Bright green
    else:
        return "#ffffff"  # White default


def get_zone_color(zone_type):
    """Get color based on zone type"""
    if "Strong Supply" in zone_type:
        return "#ff0000"  # Bright red
    elif "Moderate Supply" in zone_type:
        return "#ff6b6b"  # Medium red
    elif "Supply" in zone_type:
        return "#ff9e9e"  # Light red
    elif "Strong Demand" in zone_type:
        return "#00c700"  # Bright green
    elif "Moderate Demand" in zone_type:
        return "#5aff5a"  # Medium green
    elif "Demand" in zone_type:
        return "#9dff9d"  # Light green
    else:
        return "#ffffff"  # White default


def create_score_gauge_html(score):
    """
    Create an SVG gauge that shows a sentiment score from -100 to +100.
    Negative scores are shown in red on the left side, positive scores in green on the right.
    The needle points to the score value.
    """
    # Calculate the needle position based on the score
    # For a semicircular gauge, we need to map -100 to 180 degrees and +100 to 0 degrees
    # with 0 score at 90 degrees (pointing straight up)

    # Normalize the score to a value between 0 and 1
    normalized_score = (score + 100) / 200

    # Calculate the angle in degrees (180° for -100, 0° for +100)
    angle_degrees = 180 - (normalized_score * 180)

    # Convert angle to radians for SVG calculations
    angle_radians = angle_degrees * (3.14159 / 180)

    # Set the center point and radius of the gauge
    center_x = 80
    center_y = 50
    radius = 40

    # Calculate the position of the needle endpoint
    needle_x = center_x + radius * math.cos(angle_radians)
    needle_y = center_y - radius * \
        math.sin(angle_radians)  # Subtract to point upward

    # Create the SVG
    svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="160" height="100" viewBox="0 0 160 100">
        <!-- Background path completely removed -->
        
        <!-- Gauge color ranges: red for negative, green for positive -->
        <path d="M 20 50 A 60 60 0 0 1 80 10" stroke="#ff0000" stroke-width="10" fill="none" />
        <path d="M 80 10 A 60 60 0 0 1 140 50" stroke="#00ff00" stroke-width="10" fill="none" />
        
        <!-- Gauge needle -->
        <line x1="{center_x}" y1="{center_y}" x2="{needle_x}" y2="{needle_y}" 
              stroke="#ffffff" stroke-width="2" />
        <circle cx="{center_x}" cy="{center_y}" r="5" fill="#ffffff" />
        
        <!-- Score value displayed below the needle -->
        <text x="{center_x}" y="75" text-anchor="middle" font-size="12" fill="#ffffff">{score}</text>
        
        <!-- Scale labels -->
        <text x="20" y="65" text-anchor="middle" font-size="10" fill="#ff0000">-100</text>
        <text x="140" y="65" text-anchor="middle" font-size="10" fill="#00ff00">+100</text>
    </svg>
    '''

    return svg


def create_html_report(predictions, output_file='forex_analysis.html'):
    """Create an HTML report with dark theme and color-coded elements"""
    # Sort predictions from most bullish to most bearish
    sorted_pairs = sorted(predictions.items(),
                          key=lambda x: x[1]['score'], reverse=True)

    # HTML header with stylesheets
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Forex Analysis Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #1e1e2e;
                color: #e0e0e0;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            header {
                background-color: #2d2d44;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                text-align: center;
            }
            h1 {
                margin: 0;
                color: #ffffff;
                font-size: 28px;
            }
            .pair-card {
                background-color: #2d2d44;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .pair-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #444;
                padding-bottom: 15px;
                margin-bottom: 15px;
            }
            .pair-name {
                font-size: 24px;
                font-weight: bold;
            }
            .sentiment-badge {
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
                color: #000;
            }
            .forecast-section {
                background-color: #383852;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
            }
            .forecast-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #b8b8ff;
            }
            .price-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }
            .price-item {
                background-color: #2a2a40;
                padding: 10px;
                border-radius: 6px;
            }
            .price-label {
                font-size: 12px;
                color: #9e9eb3;
            }
            .price-value {
                font-size: 18px;
                font-weight: bold;
            }
            .zones-section {
                background-color: #383852;
                border-radius: 8px;
                padding: 15px;
            }
            .zone-item {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 10px;
            }
            .zone-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                font-weight: bold;
            }
            .zone-strategy {
                background-color: rgba(0, 0, 0, 0.2);
                padding: 8px;
                border-radius: 4px;
                font-style: italic;
                margin-top: 8px;
            }
            .gauge-container {
                display: flex;
                justify-content: center;
                margin: 10px 0;
            }
            .positive {
                color: #00ff00;
            }
            .negative {
                color: #ff5252;
            }
            .neutral {
                color: #e0e0e0;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
            }
            @media (max-width: 768px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Weekly Forex Predictions with H4 Supply and Demand Zones</h1>
                <p>Based on COT data analysis and historical performance patterns</p>
            </header>
            
            <div class="grid-container">
    '''

    # Add content for each pair
    for pair, data in sorted_pairs:
        score = data['score']
        sentiment = data['sentiment']
        sentiment_color = get_sentiment_color(sentiment)

        # Create pair card
        html += f'''
            <div class="pair-card">
                <div class="pair-header">
                    <div class="pair-name">{pair}</div>
                    <div class="sentiment-badge" style="background-color: {sentiment_color};">{sentiment}</div>
                </div>
                
                <div class="gauge-container">
                    {create_score_gauge_html(score)}
                </div>
        '''

        # Add forecast section if available
        if data['forecast']:
            forecast = data['forecast']
            # Determine if price is forecasted up or down
            price_change = forecast['Close'] - forecast['Open']
            price_color = "#52ff6a" if price_change >= 0 else "#ff5252"

            html += f'''
                <div class="forecast-section">
                    <div class="forecast-title">Weekly Forecast</div>
                    <div class="price-grid">
                        <div class="price-item">
                            <div class="price-label">Open</div>
                            <div class="price-value">{forecast['Open']}</div>
                        </div>
                        <div class="price-item">
                            <div class="price-label">High</div>
                            <div class="price-value positive">{forecast['High']}</div>
                        </div>
                        <div class="price-item">
                            <div class="price-label">Low</div>
                            <div class="price-value negative">{forecast['Low']}</div>
                        </div>
                        <div class="price-item">
                            <div class="price-label">Close</div>
                            <div class="price-value" style="color: {price_color};">{forecast['Close']}</div>
                        </div>
                    </div>
                </div>
            '''

            # Add supply and demand zones if available
            if data['supply_demand_zones']:
                zones = data['supply_demand_zones']

                html += f'''
                <div class="zones-section">
                    <div class="forecast-title">H4 Supply and Demand Zones</div>
                    <p>Weekly Bias: <span style="color: {'#52ff6a' if zones['overall_bias'] == 'Bullish' else '#ff5252'}; font-weight: bold;">{zones['overall_bias']}</span></p>
                '''

                for zone in zones['supply_demand_zones']:
                    zone_type = zone['type']
                    zone_color = get_zone_color(zone_type)

                    html += f'''
                    <div class="zone-item" style="background-color: {zone_color}20; border-left: 4px solid {zone_color};">
                        <div class="zone-header">
                            <div style="color: {zone_color};">{zone_type}</div>
                            <div>{zone['zone_low']} - {zone['zone_high']}</div>
                        </div>
                        <div>{zone['description']}</div>
                        <div class="zone-strategy">{zone['entry_strategy']}</div>
                    </div>
                    '''

                html += '</div>'
        else:
            html += '<p>Insufficient data for forecast</p>'

        html += '</div>'

    # Close the HTML document
    html += '''
            </div>
        </div>
    </body>
    </html>
    '''

    # Write the HTML to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Analysis saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Set file paths
    price_csv_path = "forex_data/all_forex_weekly.csv"
    cot_csv_path = "cot_currencies_2025.csv"
    html_output_path = "forex_analysis.html"

    # List of pairs to analyze
    pairs = [
        "EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD", "USD/CAD", "USD/CHF", "USD/JPY",
        "EUR/AUD", "EUR/CHF", "EUR/CAD", "EUR/GBP", "EUR/JPY", "EUR/NZD",
        "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/JPY", "GBP/NZD",
        "AUD/CAD", "AUD/CHF", "AUD/JPY", "AUD/NZD",
        "NZD/CAD", "NZD/CHF", "NZD/JPY"
    ]

    # Generate predictions
    predictions = generate_weekly_predictions_with_supply_demand(
        price_csv_path, cot_csv_path, pairs)

    # Create HTML report
    create_html_report(predictions, html_output_path)

    # Also print summary to console for reference
    print("Weekly Forex Predictions with H4 Supply and Demand Zones")
    print("=" * 80)

    # Print predictions in descending order (most bullish to most bearish)
    sorted_pairs = sorted(predictions.items(),
                          key=lambda x: x[1]['score'], reverse=True)

    for pair, data in sorted_pairs:
        print(f"\n{pair}: {data['score']} - {data['sentiment']}")

        if data['forecast']:
            forecast = data['forecast']
            print(
                f"  Weekly Forecast: Open: {forecast['Open']}, High: {forecast['High']}, Low: {forecast['Low']}, Close: {forecast['Close']}")

            if data['supply_demand_zones']:
                zones = data['supply_demand_zones']
                print(f"  Weekly Bias: {zones['overall_bias']}")

                print("  H4 Supply and Demand Zones:")
                for zone in zones['supply_demand_zones']:
                    print(
                        f"    {zone['type']}: {zone['zone_low']} - {zone['zone_high']}")
                    print(f"      {zone['description']}")
                    print(f"      Strategy: {zone['entry_strategy']}")
        else:
            print("  Insufficient data for forecast")
