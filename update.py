# %%
from pathlib import Path
from itertools import cycle, count

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from plotly.subplots import make_subplots

# %%
# Load S&P 500 data
csv_file = Path("sp500.csv")
df = yf.download("^GSPC")

df = df.reset_index()  # Use an integer index
df.columns = df.columns.droplevel(1)  # Remove the second level (ticker)
df.to_csv(csv_file)

df = df.rename(columns={"Date": "date", "Close": "value"})
df = df[["date", "value"]]
df["date"] = pd.to_datetime(df["date"])

# %%
# All drawdowns over threshold
threshold = -0.10

df["cummax"] = df["value"].cummax()
df["start"] = df.groupby("cummax")["date"].transform("first")
df["end"] = df.groupby("cummax")["date"].transform("last")

min_indices = df.groupby("cummax")["value"].transform("idxmin")
df["bottom"] = df.loc[min_indices, "date"].values
df["drawdown"] = (df.groupby("cummax")["value"].transform("min") - df["cummax"]) / df[
    "cummax"
]

drawdowns = df[df["drawdown"] < threshold].groupby("cummax").max().reset_index()
drawdowns = drawdowns[["start", "bottom", "end", "drawdown"]]

drawdowns["to_bottom"] = drawdowns["bottom"] - drawdowns["start"]
drawdowns["to_recovery"] = drawdowns["end"] - drawdowns["start"]

# %%
dat = drawdowns.copy()

dat["to_bottom"] = dat["to_bottom"].dt.days
dat["to_recovery"] = dat["to_recovery"].dt.days
dat["drawdown"] = dat["drawdown"].apply(lambda x: f"{x:.2%}")
dat["start"] = dat["start"].dt.strftime("%Y-%m-%d")
dat["bottom"] = dat["bottom"].dt.strftime("%Y-%m-%d")
dat["end"] = dat["end"].dt.strftime("%Y-%m-%d")

dat = dat.rename(
    columns={
        "start": "Start",
        "bottom": "Bottom",
        "end": "End",
        "drawdown": "Drawdown",
        "to_bottom": "Days To Bottom",
        "to_recovery": "Days To Recovery",
    }
)

fig = go.Figure(
    data=[
        go.Table(
            header=dict(values=list(dat.columns), align="left"),
            cells=dict(values=[dat[col] for col in dat.columns], align="right"),
        )
    ]
)
fig.update_layout(title=f"S&P 500 Drawdowns ({threshold:.0%})")
fig.write_html("docs/drawdowns.html")

# %%
fig = px.scatter(
    dat.iloc[:-1],
    title="S&P 500 Recovery Time",
    x="Days To Bottom",
    y="Days To Recovery",
    log_x=True,
    log_y=True,
)
fig.add_trace(
    go.Scatter(
        x=dat.iloc[-1:]["Days To Bottom"],
        y=dat["Days To Recovery"].iloc[-1:],
        mode="markers",
        marker=dict(size=5, color="black"),
        name=f"{dat.iloc[-1]['Start']} (so far)",
    )
)

log_x = np.log10(dat["Days To Bottom"])
log_y = np.log10(dat["Days To Recovery"])
slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
x_range = np.logspace(
    np.log10(dat["Days To Bottom"].min() * 0.9),
    np.log10(dat["Days To Bottom"].max() * 1.1),
    2,
)
residuals = log_y - (slope * log_x + intercept)
channel_width = 2 * residuals.std()
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=10 ** (intercept + slope * np.log10(x_range) + channel_width),
        mode="lines",
        name="Upper Channel",
        line=dict(color="black", width=0.5, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=10 ** (intercept + slope * np.log10(x_range) - channel_width),
        mode="lines",
        name="Lower Channel",
        line=dict(color="black", width=0.5, dash="dash"),
    )
)

fig.update_layout(
    xaxis_title="Days Until Market Bottom (log scale)",
    yaxis_title="Days Until Market Recovery (log scale)",
    hoverlabel_namelength=-1,
)

fig.write_html("docs/scatter.html")

# %%
fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"type": "scatter"}]],
    subplot_titles=("S&P 500 Drawdowns Aligned",),
    x_title="Days",
    y_title="Drawdown (%)",
)
fig.update_layout(
    hoverlabel_namelength=-1,
    yaxis_tickformat=".2%",
    margin_l=100,
)
fig.update_annotations(selector=dict(text="Drawdown (%)"), xshift=-70)

historical = drawdowns.iloc[:-1].copy()
historical = historical.sort_values(by="to_bottom", ascending=False)

for i, drawdown, color in zip(
    count(0), historical.iloc, cycle(px.colors.qualitative.Plotly)
):
    mask = (drawdown.start <= df["date"]) & (df["date"] <= drawdown.end)
    period = df[mask].copy()

    period["i"] = (period["date"] - drawdown.start).dt.days
    period["pct"] = (period["value"] - period["value"].iloc[0]) / period["value"].iloc[
        0
    ]

    legend_group = f"drawdown_{i}"
    trace_name = f"{drawdown.start.date()}, {drawdown.to_bottom.days}d bottom, {drawdown.to_recovery.days}d recovery"

    fig.add_trace(
        go.Scatter(
            x=period["i"],
            y=period["pct"],
            name=trace_name,
            mode="lines",
            line=dict(width=0.5, color=color),
            legendgroup=legend_group,
            showlegend=True,
            customdata=[drawdown.to_recovery.days],
            visible=True if drawdown.to_recovery.days < 365 * 3 else "legendonly",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[drawdown.to_bottom.days, drawdown.to_recovery.days],
            y=[drawdown.drawdown, period["pct"].iloc[-1]],
            mode="markers",
            name=trace_name,
            marker=dict(size=4, color=color),
            legendgroup=legend_group,
            showlegend=False,
            customdata=[drawdown.to_recovery.days],
            visible=True if drawdown.to_recovery.days < 365 * 3 else "legendonly",
        )
    )


drawdown = drawdowns.iloc[-1]
mask = (drawdown.start <= df["date"]) & (df["date"] <= drawdown.end)
period = df[mask].copy()

period["i"] = (period["date"] - drawdown.start).dt.days
period["pct"] = (period["value"] - period["value"].iloc[0]) / period["value"].iloc[0]

fig.add_trace(
    go.Scatter(
        x=period["i"],
        y=period["pct"],
        name=f"{drawdown.start.date()} (so far)",
        mode="lines",
        line=dict(width=1, color="black"),
    )
)

fig.update_layout(
    updatemenus=[
        dict(
            x=1.0,
            y=1.12,
            xanchor="right",
            yanchor="top",
            pad={"r": 10, "t": 10},
            direction="down",
            buttons=list(
                [
                    dict(
                        label="Hide Longer Drawdowns (> 3 years)",
                        method="update",
                        args=[
                            {
                                "visible": [
                                    True if d.customdata[0] < 365 * 3 else "legendonly"
                                    for d in fig.data[:-1]
                                ]
                                + [True]
                            }
                        ],
                    ),
                    dict(
                        label="Hide Longer Drawdowns (> 2 years)",
                        method="update",
                        args=[
                            {
                                "visible": [
                                    True if d.customdata[0] < 365 * 2 else "legendonly"
                                    for d in fig.data[:-1]
                                ]
                                + [True]
                            }
                        ],
                    ),
                    dict(
                        label="Hide Longer Drawdowns (> 1 years)",
                        method="update",
                        args=[
                            {
                                "visible": [
                                    True if d.customdata[0] < 365 * 1 else "legendonly"
                                    for d in fig.data[:-1]
                                ]
                                + [True]
                            }
                        ],
                    ),
                    dict(
                        label="Hide All Historical",
                        method="update",
                        args=[
                            {"visible": ["legendonly"] * (len(fig.data) - 1) + [True]}
                        ],
                    ),
                    dict(
                        label="Show All Historical",
                        method="update",
                        args=[{"visible": [True] * (len(fig.data) - 1) + [True]}],
                    ),
                ]
            ),
        )
    ]
)

fig.write_html("docs/aligned.html")

# %%
fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"type": "scatter"}]],
    subplot_titles=(f"S&P 500 Drawdowns Highlighted ({threshold:.0%})",),
    x_title="Days",
    y_title="Value (log scale)",
)

dat = df.copy()

for drawdown in drawdowns.iloc:
    mask = (drawdown.start <= df["date"]) & (df["date"] <= drawdown.end)
    period = dat[mask]

    fig.add_trace(
        go.Scatter(
            x=period["date"],
            y=period["value"],
            mode="lines",
            line=dict(width=1, color="red"),
            name=f"{drawdown.start.date()}, {drawdown.drawdown:.2%} max, {drawdown.to_bottom.days}d bottom, {drawdown.to_recovery.days}d recovery",
            showlegend=False,
        )
    )

    dat.loc[mask.shift(1) & mask.shift(-1), "value"] = np.nan

fig.add_trace(
    go.Scatter(
        x=dat["date"],
        y=dat["value"],
        mode="lines",
        line=dict(width=1, color="blue"),
        name="S&P 500",
        showlegend=False,
    )
)

# Set log scales on both axes
fig.update_layout(
    hoverlabel_namelength=-1,
    yaxis_type="log",
)

fig.write_html("docs/chart.html")
