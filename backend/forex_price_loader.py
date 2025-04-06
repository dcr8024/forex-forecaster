import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from config import TRADE_API_KEY


def fetch_forex_data_tradermade(api_key, forex_pairs, start_date=None, end_date=None, weeks=52):
    """
    Fetch forex data using TraderMade API (free plan: 1000 requests/month)

    Args:
        api_key: Your TraderMade API key (free tier available)
        forex_pairs: List of forex pairs to fetch
        start_date: Start date as string 'YYYY-MM-DD' or datetime object
        end_date: End date as string 'YYYY-MM-DD' or datetime object
        weeks: Number of weeks to fetch if start_date is None

    Returns:
        Combined DataFrame with all forex data
    """
    # Setup dates - end date is last week
    if end_date is None:
        end_date = datetime.now() - timedelta(days=7)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None:
        start_date = end_date - timedelta(weeks=weeks)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    print(
        f"Fetching forex data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Create directory for data if it doesn't exist
    if not os.path.exists('forex_data'):
        os.makedirs('forex_data')

    # For storing all data from all pairs
    all_data = []

    # Process each currency pair
    for pair in forex_pairs:
        name = pair[:3]+"/"+pair[3:]
        print(f"\nFetching data for {name}...")

        # Try using the daily endpoint and convert to weekly
        try:
            url = "https://marketdata.tradermade.com/api/v1/timeseries"
            params = {
                "currency": pair,
                "api_key": api_key,
                "format": "records",
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "interval": "daily"  # Get daily data and we'll convert to weekly
            }

            print(f"Making API request for {pair}...")

            # Add delay to avoid rate limits
            time.sleep(1)

            response = requests.get(url, params=params)

            # Print response status
            print(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                print(f"Error response: {response.text}")
                continue

            # Parse the response
            data = response.json()

            if "quotes" in data and data["quotes"]:
                print(
                    f"Successfully retrieved {len(data['quotes'])} daily data points")

                # Convert to DataFrame
                df = pd.DataFrame(data["quotes"])

                # Convert date strings to datetime
                df['date'] = pd.to_datetime(df['date'])

                # Group by week
                df['week'] = df['date'].dt.to_period('W')
                weekly_df = df.groupby('week').agg({
                    'date': 'last',  # Last date of the week
                    'open': 'first',  # First open price
                    'high': 'max',    # Highest high
                    'low': 'min',     # Lowest low
                    'close': 'last'   # Last close price
                }).reset_index(drop=True)

                # Format the result
                weekly_df['Pair'] = name
                weekly_df['Weekly_Return'] = weekly_df['close'].pct_change() * \
                    100

                # Rename columns to match expected format
                weekly_df = weekly_df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close'
                })

                # Add to the combined data
                all_data.append(weekly_df)
                print(f"Added {len(weekly_df)} weekly data points for {pair}")
            else:
                print("No quotes found in the response")

        except Exception as e:
            print(f"Error processing {pair}: {e}")

    # Combine all data and save to a single CSV
    if all_data:
        try:
            combined_df = pd.concat(all_data)

            # Set date as index
            combined_df = combined_df.set_index('date')

            # Save to CSV
            csv_path = os.path.join('forex_data', "all_forex_weekly.csv")
            combined_df.to_csv(csv_path)
            print(f"\nSaved combined data for all pairs to {csv_path}")
            print(f"Total rows: {len(combined_df)}")

            # Print summary
            print("\nSummary of fetched data:")
            for pair in forex_pairs:
                pair = name
                pair_data = combined_df[combined_df['Pair'] == pair]
                print(f"{pair}:")
                if pair_data.empty:
                    print(f"  No data available")
                else:
                    print(f"  Rows: {len(pair_data)}")
                    print(
                        f"  Date range: {pair_data.index.min().date()} to {pair_data.index.max().date()}")

            return combined_df
        except Exception as e:
            print(f"Error creating combined CSV: {e}")
            return pd.DataFrame()
    else:
        print("No data was successfully retrieved")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Your TraderMade API key
    api_key = TRADE_API_KEY

    # All forex pairs from the images
    forex_pairs = [
        # USD crosses
        "USDCAD", "USDCHF", "USDJPY",

        # EUR crosses
        "EURUSD", "EURAUD", "EURCHF", "EURCAD", "EURGBP", "EURNZD", "EURJPY",

        # GBP crosses
        "GBPAUD", "GBPJPY", "GBPNZD", "GBPUSD", "GBPCAD", "GBPCHF",

        # AUD crosses
        "AUDNZD", "AUDCAD", "AUDJPY", "AUDCHF", "AUDUSD",

        # NZD crosses
        "NZDUSD", "NZDJPY", "NZDCAD", "NZDCHF"
    ]

    # Set a date range (end date is one week ago)
    end_date = datetime.now() - timedelta(days=7)
    start_date = end_date - timedelta(weeks=52)  # One year before end date

    # Fetch the data and save only the combined CSV
    forex_data = fetch_forex_data_tradermade(
        api_key, forex_pairs, start_date, end_date)
