from pathlib import Path

import yfinance as yf
import pandas as pd

csv_file = Path("sp500.csv")
if not csv_file.exists():
    df = yf.download("^GSPC")

    df = df.reset_index()  # Use an integer index
    df.columns = df.columns.droplevel(1)  # Remove the second level (ticker)
    df.to_csv(csv_file)
else:
    df = pd.read_csv(csv_file, parse_dates=True)

# Calculate the running maximum price
df["Running Max"] = df["Close"].cummax()
df["Drawdown"] = (df["Close"] - df["Running Max"]) / df["Running Max"]

downturn_days = df[df["Drawdown"] <= -0.10]
downturn_days = downturn_days[["Date", "Close", "Running Max", "Drawdown"]]

print("Days with more than 10% downturn from peak:")
print(downturn_days)
