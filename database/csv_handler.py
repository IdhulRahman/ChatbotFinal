import pandas as pd

def load_csv(file_path):
    """
    Load the CSV file and return a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        DataFrame: Loaded data.
    """
    return pd.read_csv(file_path, delimiter=',')

def search_csv(query, csv_data):
    """
    Search the CSV for a query in the 'Name' or 'ISBN' columns.
    Args:
        query (str): The search term provided by the user.
        csv_data (DataFrame): The loaded CSV data as a pandas DataFrame.
    Returns:
        list: Matching rows as dictionaries.
    """
    results = csv_data[
        csv_data['Name'].str.contains(query, case=False, na=False) |
        csv_data['ISBN'].astype(str).str.contains(query, na=False)
    ]
    return results.to_dict(orient="records")
