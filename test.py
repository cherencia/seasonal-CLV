from beta_geo_seasonal import BetaGeoModelWithSeasonality
from summary_functions import summary_data_from_transaction_data_season 
import pandas as pd
import sqlalchemy

# Updated connection string using ODBC Driver 17 for SQL Server
connection_string = (
    'mssql+pyodbc://@Gpdwhdb/Cubeware_OLAP?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
)

# Create an engine
engine = sqlalchemy.create_engine(connection_string)
# Execute the SQL query and fetch the result into a DataFrame
select_query = """
SELECT 
    kunden.FK_KundenID AS ID,
    kunden.FK_NeuanlageID AS Datum,
    kunden.Flag_Hoch_Saison as high_season,
    (kunden.Nachfrage_Artikel_Wert - kunden.Nachfrage_Artikel_EK) AS monetary_value
FROM 
    [Cubeware_OLAP].[DIM].[Auftrag] kunden
WHERE 
    kunden.FK_VertriebswegID = 2 
    AND kunden.FK_AuftragsherkunftID != 7
    AND kunden.FK_AuftragsstatusID != 8
    AND YEAR(kunden.FK_NeuanlageID) >= 2021
ORDER BY
    kunden.FK_NeuanlageID


"""
transactions = pd.read_sql(select_query, engine)



# Display the result DataFrame
transactions

summary = summary_data_from_transaction_data_season(

    transactions,
    customer_id_col='ID',
    datetime_col='Datum',
    monetary_value_col='monetary_value',
    high_season_col='high_season',
    datetime_format='%Y-%m-%d',
    observation_period_end='2023-12-31',
    freq='D',
    include_first_transaction=True
)

# Rename the high_season_sum column back to high_season
summary.rename(columns={'high_season_tx_sum': 'high_season'}, inplace=True)

model = BetaGeoModelWithSeasonality(
    data=summary,
    model_config={
        "r_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "alpha_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "a_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "b_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "phi_prior": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
    },
    sampler_config={
        "draws": 1000,
        "tune": 1000,
        "chains": 2,
        "cores": 2,
    },
)
model.fit()
print(model.fit_summary())

expected_purchases = model.expected_purchases(future_t=10)
probability_alive = model.expected_probability_alive()
expected_purchases_new_customer = model.expected_purchases_new_customer(t=10)