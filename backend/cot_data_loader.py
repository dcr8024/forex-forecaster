import os
import pandas as pd
import requests
import zipfile
import io
from datetime import date, datetime


def cot_year_currencies(year=datetime.now().year, include_previous_year=True, currencies=None, save_csv=True, csv_filename=None, store_txt=False, max_weeks=52):
    '''Downloads the legacy futures only COT report data for a single year and filters for specific currencies.

    Args:
        year (int): Year to download data for, as YYYY. Defaults to 2025.
        include_previous_year (bool): Whether to also download data for the previous year. Defaults to True.
        currencies (list): List of currency codes to filter by. Defaults to major currencies.
        save_csv (bool): Whether to save the data as CSV. Defaults to True.
        csv_filename (str): Name of the CSV file. If None, defaults to "cot_currencies_{year}.csv".
        store_txt (bool): Whether to store the extracted text file. Defaults to False.
        max_weeks (int): Maximum number of newest weeks to include for each currency. Defaults to 52.

    Returns:
        A DataFrame with the COT data for the specified year and currencies, and tuple with record count.
    '''
    # Default currencies if none provided
    if currencies is None:
        currencies = ['EUR', 'JPY', 'NZD', 'AUD', 'GBP', 'CHF', 'CAD']

    # List to store DataFrames for each year
    dfs = []

    # Years to download
    years_to_download = [year]
    if include_previous_year:
        years_to_download.append(year - 1)

    # Sort years (previous year first, then current)
    years_to_download.sort()

    # Download data for each year
    for yr in years_to_download:
        # Download and extract data
        rep = "deacot"
        # Use year-specific filename to avoid conflicts
        txt = f"annual_{yr}.txt"
        cot_url = "https://cftc.gov/files/dea/history/" + \
            rep + str(yr) + ".zip"

        try:
            r = requests.get(cot_url)
            # Check if request was successful
            if r.status_code != 200:
                print(
                    f"Warning: Could not download data for year {yr}. Status code: {r.status_code}")
                continue

            z = zipfile.ZipFile(io.BytesIO(r.content))

            # Get the list of files in the zip
            extracted_files = z.namelist()
            if not extracted_files:
                print(f"Error: Zip file for year {yr} is empty.")
                continue

            # Extract to a year-specific directory to avoid conflicts
            year_dir = f"cot_temp_{yr}"
            os.makedirs(year_dir, exist_ok=True)
            z.extractall(year_dir)

            # Find the extracted file (usually annual.txt)
            extracted_file_path = None
            for filename in extracted_files:
                full_path = os.path.join(year_dir, filename)
                if os.path.exists(full_path):
                    extracted_file_path = full_path
                    break

            if not extracted_file_path:
                print(f"Error: Could not find extracted file for year {yr}.")
                continue

            # Read the data
            df_year = pd.read_csv(extracted_file_path, low_memory=False)
            print(f"Downloaded {len(df_year)} records for year {yr}.")

            # Filter for currencies
            filtered_df_year = pd.DataFrame()

            # Check common column names
            possible_columns = [
                'MARKET_AND_EXCHANGE_NAMES', 'COMMODITY_NAME', 'Contract_Market_Name',
                'Market_and_Exchange_Names', 'Market and Exchange Names',
                'CFTC_Contract_Market_Code', 'Commodity', 'Name'
            ]

            # Find which columns exist in the DataFrame
            existing_columns = [
                col for col in possible_columns if col in df_year.columns]

            # Try filtering using existing columns with improved logic to handle slashes
            for col in existing_columns:
                try:
                    mask = pd.Series(False, index=df_year.index)
                    for currency_term in currencies:
                        # Match exact currency terms and exclude rows with slashes
                        exact_match = ((df_year[col].str.strip().str.upper() == currency_term.upper()) |
                                       (df_year[col].str.contains(f'^{currency_term}$', case=False, regex=True)))

                        # If we're matching a broader term, make sure it's not part of a pair (no slash)
                        contains_match = (df_year[col].str.contains(f'\\b{currency_term}\\b',
                                                                    case=False, na=False, regex=True) &
                                          ~df_year[col].str.contains('/', na=False))

                        mask = mask | exact_match | contains_match

                    temp_df = df_year[mask]
                    if len(temp_df) > 0:
                        filtered_df_year = temp_df
                        print(
                            f"Found {len(filtered_df_year)} currency matches for year {yr}.")
                        break
                except Exception as e:
                    print(f"Error in filtering column {col}: {str(e)}")
                    continue

            # If no matches found, try all string columns
            if filtered_df_year.empty:
                for col in df_year.columns:
                    if df_year[col].dtype == 'object':
                        try:
                            mask = pd.Series(False, index=df_year.index)
                            for currency_term in currencies:
                                # Only match if not part of a currency pair (no slash)
                                mask = mask | (df_year[col].str.contains(
                                    f'\\b{currency_term}\\b', case=False, na=False, regex=True) &
                                    ~df_year[col].str.contains('/', na=False))

                            temp_df = df_year[mask]
                            if len(temp_df) > 0:
                                filtered_df_year = temp_df
                                print(
                                    f"Found {len(filtered_df_year)} currency matches for year {yr} (using column {col}).")
                                break
                        except:
                            continue

            # Add to list if we found matches
            if not filtered_df_year.empty:
                dfs.append(filtered_df_year)

            # Clean up the temporary directory
            if not store_txt:
                try:
                    # Remove the file
                    if extracted_file_path and os.path.exists(extracted_file_path):
                        os.remove(extracted_file_path)

                    # Remove the directory
                    if os.path.exists(year_dir):
                        os.rmdir(year_dir)
                except Exception as e:
                    print(
                        f"Warning: Could not clean up temporary files for year {yr}: {str(e)}")

        except Exception as e:
            print(f"Error processing year {yr}: {str(e)}")

    # Combine data from all years
    if dfs:
        filtered_df = pd.concat(dfs, ignore_index=True)
        print(
            f"Combined data contains {len(filtered_df)} records across {len(dfs)} years.")
    else:
        filtered_df = pd.DataFrame()
        print("Warning: No matching data found for any of the requested years.")
        return filtered_df, 0

    # Generate CSV filename
    if csv_filename is None:
        csv_filename = f"cot_currencies_{year}.csv"

    # Save CSV if requested
    if save_csv and not filtered_df.empty:
        # Select only the columns you need for positions
        position_columns = [
            'Market_Name',  # Already renamed or original name
            'Date',         # Already renamed or original name
            'Comm_Long',    # Commercial long position (current)
            'Comm_Short',   # Commercial short position (current)
            # Retail long position (current)
            'Nonreportable Positions-Long (All)',
            # Retail short position (current)
            'Nonreportable Positions-Short (All)',
        ]

        # Create mapping for column renaming
        column_map = {
            # Date columns (try different variations)
            "As of Date in Form YYYY-MM-DD": "Date",
            "Report_Date_as_YYYY-MM-DD": "Date",
            "As of Date in Form YYMMDD": "Date_YYMMDD",

            # Market identification column
            "Market and Exchange Names": "Market_Name",
            "MARKET_AND_EXCHANGE_NAMES": "Market_Name",
            "arket_Name": "Market_Name",  # Fix for the typo in your headers

            # Commercial positions
            "Commercial Positions-Long (All)": "Comm_Long",
            "Commercial Positions-Short (All)": "Comm_Short",

            # Nonreportable (retail) positions
            "Nonreportable Positions-Long (All)": "Retail_Long",
            "Nonreportable Positions-Short (All)": "Retail_Short"
        }

        # Only rename columns that exist in the DataFrame
        columns_to_rename = {
            k: v for k, v in column_map.items() if k in filtered_df.columns}

        if columns_to_rename:
            filtered_df = filtered_df.rename(columns=columns_to_rename)

        # Now ensure we have the correctly renamed columns
        actual_columns = []

        # Market and Date columns
        if 'Market_Name' in filtered_df.columns:
            actual_columns.append('Market_Name')
        elif 'arket_Name' in filtered_df.columns:
            actual_columns.append('arket_Name')

        if 'Date' in filtered_df.columns:
            actual_columns.append('Date')

        # Commercial positions
        if 'Comm_Long' in filtered_df.columns:
            actual_columns.append('Comm_Long')
        elif 'Commercial Positions-Long (All)' in filtered_df.columns:
            actual_columns.append('Commercial Positions-Long (All)')

        if 'Comm_Short' in filtered_df.columns:
            actual_columns.append('Comm_Short')
        elif 'Commercial Positions-Short (All)' in filtered_df.columns:
            actual_columns.append('Commercial Positions-Short (All)')

        # Retail positions
        if 'Retail_Long' in filtered_df.columns:
            actual_columns.append('Retail_Long')
        elif 'Nonreportable Positions-Long (All)' in filtered_df.columns:
            actual_columns.append('Nonreportable Positions-Long (All)')

        if 'Retail_Short' in filtered_df.columns:
            actual_columns.append('Retail_Short')
        elif 'Nonreportable Positions-Short (All)' in filtered_df.columns:
            actual_columns.append('Nonreportable Positions-Short (All)')

        # Filter DataFrame to only include the columns we want
        if actual_columns:
            filtered_df = filtered_df[actual_columns]
            print(f"Selected {len(actual_columns)} columns: {actual_columns}")
        else:
            print("Warning: Could not identify specific columns to keep.")
            print("Keeping all columns in the CSV file.")

        # Calculate percentage changes (week-over-week)
        if len(filtered_df) > 0:
            # Sort by market and date to ensure proper calculation
            if 'Date' in filtered_df.columns and ('Market_Name' in filtered_df.columns or 'arket_Name' in filtered_df.columns):
                market_col = 'Market_Name' if 'Market_Name' in filtered_df.columns else 'arket_Name'
                filtered_df = filtered_df.sort_values(by=[market_col, 'Date'])

                # Get the current column names
                comm_long_col = 'Comm_Long' if 'Comm_Long' in filtered_df.columns else 'Commercial Positions-Long (All)'
                comm_short_col = 'Comm_Short' if 'Comm_Short' in filtered_df.columns else 'Commercial Positions-Short (All)'
                retail_long_col = 'Retail_Long' if 'Retail_Long' in filtered_df.columns else 'Nonreportable Positions-Long (All)'
                retail_short_col = 'Retail_Short' if 'Retail_Short' in filtered_df.columns else 'Nonreportable Positions-Short (All)'

                # Get the change column names (for option 3 - use change values for first records)
                comm_long_change_col = next((col for col in filtered_df.columns if "COMM" in col.upper(
                ) and "LONG" in col.upper() and "CHANGE" in col.upper()), None)
                comm_short_change_col = next((col for col in filtered_df.columns if "COMM" in col.upper(
                ) and "SHORT" in col.upper() and "CHANGE" in col.upper()), None)
                retail_long_change_col = next((col for col in filtered_df.columns if ("NONREPT" in col.upper(
                ) or "NONREPORTABLE" in col.upper()) and "LONG" in col.upper() and "CHANGE" in col.upper()), None)
                retail_short_change_col = next((col for col in filtered_df.columns if ("NONREPT" in col.upper(
                ) or "NONREPORTABLE" in col.upper()) and "SHORT" in col.upper() and "CHANGE" in col.upper()), None)

                # Calculate percentage changes by group
                filtered_df['Comm_Long_Pct_Change'] = filtered_df.groupby(
                    market_col)[comm_long_col].pct_change() * 100
                filtered_df['Comm_Short_Pct_Change'] = filtered_df.groupby(
                    market_col)[comm_short_col].pct_change() * 100
                filtered_df['Retail_Long_Pct_Change'] = filtered_df.groupby(
                    market_col)[retail_long_col].pct_change() * 100
                filtered_df['Retail_Short_Pct_Change'] = filtered_df.groupby(
                    market_col)[retail_short_col].pct_change() * 100

                # For first records of each group, replace NaN with calculation using absolute change values
                if all(col is not None for col in [comm_long_change_col, comm_short_change_col, retail_long_change_col, retail_short_change_col]):
                    print(
                        "Using absolute change values to calculate percentage changes for first records of each currency")

                    # Identify first records for each market
                    first_records = filtered_df.groupby(
                        market_col).first().index.tolist()

                    # For each market's first record, calculate percentage change based on absolute change
                    for market in first_records:
                        mask = (filtered_df[market_col] == market) & (
                            filtered_df['Comm_Long_Pct_Change'].isna())

                        # Safe division - handle cases where position is zero to avoid division by zero
                        filtered_df.loc[mask, 'Comm_Long_Pct_Change'] = filtered_df.loc[mask].apply(
                            lambda row: (
                                row[comm_long_change_col] / row[comm_long_col] * 100)
                            if row[comm_long_col] != 0 else 0, axis=1)

                        filtered_df.loc[mask, 'Comm_Short_Pct_Change'] = filtered_df.loc[mask].apply(
                            lambda row: (
                                row[comm_short_change_col] / row[comm_short_col] * 100)
                            if row[comm_short_col] != 0 else 0, axis=1)

                        filtered_df.loc[mask, 'Retail_Long_Pct_Change'] = filtered_df.loc[mask].apply(
                            lambda row: (
                                row[retail_long_change_col] / row[retail_long_col] * 100)
                            if row[retail_long_col] != 0 else 0, axis=1)

                        filtered_df.loc[mask, 'Retail_Short_Pct_Change'] = filtered_df.loc[mask].apply(
                            lambda row: (
                                row[retail_short_change_col] / row[retail_short_col] * 100)
                            if row[retail_short_col] != 0 else 0, axis=1)
                else:
                    print(
                        "Warning: Could not find change columns to calculate percentage changes for first records")
                    # Fill NaN values with zero for first records
                    filtered_df['Comm_Long_Pct_Change'] = filtered_df['Comm_Long_Pct_Change'].fillna(
                        0)
                    filtered_df['Comm_Short_Pct_Change'] = filtered_df['Comm_Short_Pct_Change'].fillna(
                        0)
                    filtered_df['Retail_Long_Pct_Change'] = filtered_df['Retail_Long_Pct_Change'].fillna(
                        0)
                    filtered_df['Retail_Short_Pct_Change'] = filtered_df['Retail_Short_Pct_Change'].fillna(
                        0)

                # Round to 2 decimal places
                filtered_df['Comm_Long_Pct_Change'] = filtered_df['Comm_Long_Pct_Change'].round(
                    2)
                filtered_df['Comm_Short_Pct_Change'] = filtered_df['Comm_Short_Pct_Change'].round(
                    2)
                filtered_df['Retail_Long_Pct_Change'] = filtered_df['Retail_Long_Pct_Change'].round(
                    2)
                filtered_df['Retail_Short_Pct_Change'] = filtered_df['Retail_Short_Pct_Change'].round(
                    2)

                print("Added percentage change columns.")
            else:
                print(
                    "Warning: Could not calculate percentage changes due to missing Date or Market_Name columns.")

        # *** IMPORTANT ADDITION: Filter to keep only the 52 newest entries per currency ***
        if 'Date' in filtered_df.columns and ('Market_Name' in filtered_df.columns or 'arket_Name' in filtered_df.columns):
            market_col = 'Market_Name' if 'Market_Name' in filtered_df.columns else 'arket_Name'

            # Convert Date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['Date']):
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

            # Group by market, sort by date (newest first), and keep only max_weeks entries
            filtered_df = filtered_df.sort_values(['Date'], ascending=False)

            # Create a list to hold filtered data for each currency
            newest_records = []

            # For each unique market/currency
            for market, group in filtered_df.groupby(market_col):
                # Sort by date descending and take only the first max_weeks records
                newest_group = group.sort_values(
                    'Date', ascending=False).head(max_weeks)
                newest_records.append(newest_group)

            # Combine the filtered groups
            if newest_records:
                filtered_df = pd.concat(newest_records, ignore_index=True)
                # Resort by market and date (oldest first) for final output
                filtered_df = filtered_df.sort_values([market_col, 'Date'])
                print(
                    f"Filtered to include only the {max_weeks} newest entries for each currency.")
            else:
                print("Warning: No data remaining after filtering for newest entries.")
        else:
            print(
                f"Warning: Could not filter to {max_weeks} newest entries due to missing Date or Market_Name columns.")

        # Save the filtered DataFrame to CSV
        filtered_df.to_csv(csv_filename, index=False)
        print(f"Saved filtered data to {csv_filename}")

    # Return filtered DataFrame and count
    return filtered_df, len(filtered_df)


# Example usage:
if __name__ == "__main__":
    # Specify the year you want to download
    year_to_download = datetime.now().year

    # Updated currency mappings to ensure precise matching
    # Using more specific full names to avoid matching currency pairs
    currency_mappings = {
        'EUR': ['EURO FX - CHICAGO MERCANTILE EXCHANGE', 'EURO FX'],
        'JPY': ['JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE', 'JAPANESE YEN'],
        'NZD': ['NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE', 'NEW ZEALAND DOLLAR'],
        'AUD': ['AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE', 'AUSTRALIAN DOLLAR'],
        'GBP': ['BRITISH POUND - CHICAGO MERCANTILE EXCHANGE', 'BRITISH POUND'],
        'CHF': ['SWISS FRANC - CHICAGO MERCANTILE EXCHANGE', 'SWISS FRANC'],
        'CAD': ['CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE', 'CANADIAN DOLLAR']
    }

    # Flatten currency list with exact matching
    all_currency_terms = []
    for terms in currency_mappings.values():
        all_currency_terms.extend(terms)

    # Download and filter data (including previous year)
    df, record_count = cot_year_currencies(
        year=year_to_download,
        include_previous_year=True,  # Include previous year's data
        currencies=all_currency_terms,
        save_csv=True,
        csv_filename=f"cot_currencies_{year_to_download}.csv",
        max_weeks=53  # Explicitly set the max weeks parameter to 52
    )

    # Print only the final record count
    print(f"Final dataset contains {record_count} records.")
