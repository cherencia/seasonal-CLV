import pandas as pd
import numpy as np
import json

import pandas as pd
import dask.dataframe as dd
from tqdm.dask import TqdmCallback  # Import TqdmCallback for Dask

def _find_first_transactions_season(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    high_season_col=None,  # New parameter for high season indicator
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    include_first_transaction=False,  # New parameter
):
    """
    Return dataframe with first transactions and count of high season repeated transactions.

    Parameters
    ----------
    transactions: :obj:`DataFrame`
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id.
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
    high_season_col: string, optional
        the column in transactions that denotes whether a transaction occurred in high season.
    observation_period_end: :obj:`datetime`
        a string or datetime to denote the final date of the study.
    datetime_format: string, optional
        a string that represents the timestamp format.
    freq: string, optional
        'D' for days, or other numpy datetime64 time units.
    """

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    if type(observation_period_end) == pd.Period:
        observation_period_end = observation_period_end.to_timestamp()

    select_columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        select_columns.append(monetary_value_col)
    if high_season_col:
        select_columns.append(high_season_col)

    transactions = transactions[select_columns].sort_values([datetime_col, customer_id_col]).copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.loc[transactions[datetime_col] <= observation_period_end]

    transactions['period'] = transactions[datetime_col].dt.to_period(freq)

    # Mark first transactions
    transactions['first'] = transactions.groupby(customer_id_col)[datetime_col].transform('min') == transactions[datetime_col]

    # Initialize high season transaction counts to zero
    transactions['high_season_tx'] = 0

    if high_season_col:
        if include_first_transaction:
            # Count high season for all transactions, including the first one
            transactions.loc[transactions[high_season_col] == 1, 'high_season_tx'] = 1
        else:
            # Only count as high season if it's not the first transaction
            transactions.loc[~transactions['first'] & (transactions[high_season_col] == 1), 'high_season_tx'] = 1

    aggregation_functions = {
        'first': 'max',  # To identify first transactions
        'high_season_tx': 'sum',  # Sum high season transactions
    }
    if monetary_value_col:
        aggregation_functions[monetary_value_col] = 'sum'  # Aggregate monetary value if provided

    aggregated_data = transactions.groupby([customer_id_col, 'period'], as_index=False).agg(aggregation_functions)

    # Reset 'datetime_col' to reflect the aggregated period
    aggregated_data[datetime_col] = aggregated_data['period'].apply(lambda x: x.start_time)
    aggregated_data.drop(columns=['period'], inplace=True)

    return aggregated_data



def summary_data_from_transaction_data_season(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    high_season_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    freq_multiplier=1,
    include_first_transaction=False,
):
    """
    Summarize transaction data with seasonal adjustment.

    This function takes a DataFrame of transaction data and aggregates it to provide
    key metrics for each customer, including transaction frequency, recency, monetary value,
    and the number of transactions during the high season.

    Parameters
    ----------
    transactions: pd.DataFrame
        A Pandas DataFrame containing the transaction data.
    customer_id_col: str
        The column in the transactions DataFrame that denotes the customer ID.
    datetime_col: str
        The column in the transactions DataFrame that denotes the datetime of the transaction.
    monetary_value_col: str, optional
        The column in the transactions DataFrame that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    high_season_col: str, optional
        The column in the transactions DataFrame that indicates whether the transaction occurred
        during the high season.
    datetime_format: str, optional
        A string that represents the datetime format. Useful if Pandas cannot infer the format.
    observation_period_end: str or datetime-like, optional
        A string or datetime to denote the final date of the observation period.
        Events after this date are truncated. If not given, defaults to the max date in `datetime_col`.
    freq: str, optional
        The frequency of the transactions (e.g., 'D' for daily). Default is 'D'.
    freq_multiplier: int, optional
        A multiplier for the frequency to adjust the time periods. Default is 1.
    include_first_transaction: bool, optional
        Whether to include the first transaction in the frequency count. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summarized transaction data for each customer.
    """
    if observation_period_end is None:
        observation_period_end = pd.to_datetime(transactions[datetime_col].max(), format=datetime_format).to_period(freq).to_timestamp()
    else:
        observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq).to_timestamp()

    repeated_transactions = _find_first_transactions_season(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        high_season_col,  # Pass the high season column
        datetime_format,
        observation_period_end,
        freq,
        include_first_transaction  # Pass include_first_transaction flag
    )

    # Prepare aggregation
    agg_dict = {
        datetime_col: ['min', 'max', 'count'],
    }
    if high_season_col:
        agg_dict["high_season_tx"] = 'sum'
    if monetary_value_col:
        agg_dict[monetary_value_col] = 'mean'

    # Group by customer and aggregate data
    customers = repeated_transactions.groupby(customer_id_col, sort=False).agg(agg_dict)

    # Flatten the MultiIndex columns created by agg
    customers.columns = ['_'.join(col).strip('_') for col in customers.columns.values]

    # Adjust frequency for the inclusion or exclusion of the first transaction
    customers["frequency"] = customers[datetime_col + "_count"] - 1 if not include_first_transaction else customers[datetime_col + "_count"]

    # Calculate T and recency
    customers["T"] = (observation_period_end - pd.to_datetime(customers[datetime_col + "_min"])).dt.days / freq_multiplier
    customers["recency"] = (pd.to_datetime(customers[datetime_col + "_max"]) - pd.to_datetime(customers[datetime_col + "_min"])).dt.days / freq_multiplier

    # Include monetary_value if specified
    if monetary_value_col:
        customers.rename(columns={f'{monetary_value_col}_mean': 'monetary_value'}, inplace=True)

    # Reset index to ensure customer_id_col is a column
    customers.reset_index(inplace=True)

    # Fill NaNs with zeros, particularly for high_season_tx if high_season_col wasn't provided
    customers.fillna(0, inplace=True)

    # Ensure columns are in the correct data type
    type_cast_dict = {'frequency': 'int', 'T': 'float', 'recency': 'float'}
    if 'high_season_tx' in customers.columns:
        type_cast_dict['high_season_tx'] = 'int'
    if monetary_value_col and 'monetary_value' in customers.columns:
        type_cast_dict['monetary_value'] = 'float'
    customers = customers.astype(type_cast_dict)

    return customers