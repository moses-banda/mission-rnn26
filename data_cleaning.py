import yfinance as yf
import pandas as pd
import numpy as np

def get_historical_data(company_list, start_date, end_date):
    nasdaq_data = pd.read_csv("NASDAQ-COMPOSITE.csv")
    nasdaq_data["Date"] = pd.to_datetime(nasdaq_data["Date"])
    nasdaq_data = nasdaq_data.set_index("Date")
    nasdaq_data.index = nasdaq_data.index.normalize()

    nasdaq_cols = {
        "Open": "Open_nasdaq",
        "High": "High_nasdaq",
        "Low": "Low_nasdaq",
        "Close": "Close_nasdaq"
    }

    for col in nasdaq_cols:
        nasdaq_data[col] = pd.to_numeric(
            nasdaq_data[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    nasdaq_data = nasdaq_data.rename(columns=nasdaq_cols)

    stock_info = []

    for company in company_list:
        ticker = yf.Ticker(company)
        company_history = ticker.history(
            start=start_date, end=end_date
        )[["Open", "High", "Low", "Close", "Volume"]]

        company_history.index = (
            company_history.index.tz_localize(None).normalize()
        )

        combined = company_history.join(nasdaq_data, how="left")

        combined["Company"] = company
        combined = combined.set_index("Company", append=True)
        combined = combined.reorder_levels(["Company", "Date"])

        stock_info.append(combined)

    stock_data = pd.concat(stock_info).sort_index()
    return stock_data

company_list = ["NVDA", "GOOGL", "META", "AAPL", "DELL", "MSFT", "TSLA", "INTC", "TXN"]

start_date = "2023-01-03"
end_date = "2025-12-30"

stock_data = get_historical_data(company_list, start_date, end_date)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data = stock_data.reset_index()
data = data.sort_values(["Date", "Company"]).copy()

feature_cols = ["Open", "High", "Low", "Close", "Volume", "Open_nasdaq", "High_nasdaq", "Low_nasdaq", "Close_nasdaq"]

data["Target"] = data.groupby("Company")["Close"].shift(-1)
data["TargetDate"] = data.groupby("Company")["Date"].shift(-1)

unique_dates = np.sort(data["Date"].unique())
split_index = int(len(unique_dates) * 0.80)
test_start_date = unique_dates[split_index]

train_mask = ((data["Date"] < test_start_date) & (data["TargetDate"] < test_start_date))
test_mask = data["Date"] >= test_start_date

train_data = data.loc[train_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[feature_cols]
X_test = test_data[feature_cols]

y_train = train_data["Target"].to_numpy()
y_test = test_data["Target"].to_numpy()

company_train = train_data["Company"].to_numpy()
company_test = test_data["Company"].to_numpy()

date_train = train_data["Date"].to_numpy()
date_test = test_data["Date"].to_numpy()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=4)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_cols = [f"PC{i}" for i in range(1, 5)]

normalized_stock_info = pd.concat([
    pd.DataFrame(
        X_train_pca,
        columns=pca_cols,
        index=train_data.index
    ),
    pd.DataFrame(
        X_test_pca,
        columns=pca_cols,
        index=test_data.index
    )
]).sort_index()

normalized_stock_info["Company"] = data.loc[
    normalized_stock_info.index, "Company"
]

print("Training shape:", X_train_pca.shape)
print("Testing shape:", X_test_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print(normalized_stock_info)

x = normalized_stock_info.to_csv('training_validation_data.csv')
