import json
import math
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]
PERIOD_OPTIONS = ["5d", "1mo", "3mo", "6mo", "1y"]
INTERVAL_OPTIONS = ["5m", "15m", "30m", "1h", "1d"]


st.set_page_config(
    page_title="Multi-Stocks AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f6f7fb; }
    [data-testid="stSidebar"] { background: #eef2f7; }
    .main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 12px 14px;
        color: #1f2937;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetric"] label { color: #64748b !important; }
    [data-testid="stMetricValue"] { color: #1f2937 !important; }
    [data-testid="stMetricDelta"] { background: #ecfdf5; border-radius: 999px; padding: 2px 8px; }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .signal-buy { color: #10b981; font-weight: 700; }
    .signal-sell { color: #ef4444; font-weight: 700; }
    .signal-watch { color: #f59e0b; font-weight: 700; }
    .app-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }
    .app-subtitle { color: #64748b; margin-bottom: 1.1rem; }
    @media (max-width: 760px) {
        .app-title { font-size: 1.35rem; }
        [data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(how="all")


@st.cache_data(ttl=180, show_spinner=False)
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    data = normalize_columns(data)
    if data.empty:
        return data
    data.index = pd.to_datetime(data.index)
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Typical"] = (data["High"] + data["Low"] + data["Close"]) / 3
    cumulative_volume = data["Volume"].replace(0, np.nan).cumsum()
    data["VWAP"] = (data["Typical"] * data["Volume"]).cumsum() / cumulative_volume

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Volume_MA"] = data["Volume"].rolling(20, min_periods=5).mean()
    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(20, min_periods=5).std() * np.sqrt(252)
    data["SMA_10"] = data["Close"].rolling(10, min_periods=5).mean()
    data["SMA_20"] = data["Close"].rolling(20, min_periods=10).mean()
    data["Momentum_5"] = data["Close"].pct_change(5)
    return data


def sigmoid(value: pd.Series | float) -> pd.Series | float:
    return 1 / (1 + np.exp(-np.clip(value, -50, 50)))


def ensemble_probabilities(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    volume_ratio = (data["Volume"] / data["Volume_MA"]).replace([np.inf, -np.inf], np.nan).fillna(1)
    trend_model = sigmoid(
        8 * data["Momentum_5"].fillna(0)
        + 2.5 * ((data["Close"] / data["SMA_20"]) - 1).replace([np.inf, -np.inf], 0).fillna(0)
        + 1.5 * ((data["Close"] / data["VWAP"]) - 1).replace([np.inf, -np.inf], 0).fillna(0)
    )
    rsi_model = np.where(
        data["RSI"].between(45, 68),
        0.62,
        np.where(data["RSI"] < 35, 0.56, np.where(data["RSI"] > 74, 0.34, 0.48)),
    )
    flow_model = sigmoid(
        1.2 * (volume_ratio - 1)
        + 1.8 * ((data["Close"] / data["VWAP"]) - 1).replace([np.inf, -np.inf], 0).fillna(0)
    )
    probability = 0.45 * trend_model + 0.30 * pd.Series(rsi_model, index=data.index) + 0.25 * flow_model
    return probability.clip(0.05, 0.95)


def ensemble_accuracy(df: pd.DataFrame, threshold: float) -> tuple[float, int]:
    data = df.copy().dropna(subset=["Close", "VWAP", "RSI", "Volume_MA", "SMA_20", "Momentum_5"])
    if len(data) < 30:
        return 0.0, 0
    probability = ensemble_probabilities(data)
    predicted_up = probability >= threshold
    actual_up = data["Close"].shift(-1) > data["Close"]
    comparable = actual_up.dropna().index.intersection(predicted_up.index)
    if len(comparable) < 10:
        return 0.0, 0
    accuracy = (predicted_up.loc[comparable] == actual_up.loc[comparable]).mean() * 100
    return float(accuracy), int(len(comparable))


def bot_action(row: pd.Series, probability: float, threshold: float) -> str:
    if probability >= threshold and row["Close"] >= row["VWAP"] and 42 <= row["RSI"] <= 72:
        return "AUTO ENTRY"
    if probability <= 0.42 or row["Close"] < row["VWAP"] or row["RSI"] >= 76:
        return "AUTO EXIT"
    return "HOLD"


def predict_next_close(df: pd.DataFrame) -> tuple[float, float]:
    clean = df[["Close", "VWAP", "RSI", "Volume"]].dropna().tail(80)
    if len(clean) < 20:
        latest = float(df["Close"].dropna().iloc[-1])
        return latest, 0.0

    y = clean["Close"].shift(-1).dropna()
    x = clean.loc[y.index, ["Close", "VWAP", "RSI", "Volume"]].copy()
    x["Volume"] = np.log1p(x["Volume"])
    x_mean = x.mean()
    x_std = x.std(ddof=0).replace(0, 1)
    normalized_x = (x - x_mean) / x_std
    design = np.column_stack([np.ones(len(normalized_x)), normalized_x.to_numpy()])
    coeffs = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]

    latest_x = clean[["Close", "VWAP", "RSI", "Volume"]].tail(1).copy()
    latest_x["Volume"] = np.log1p(latest_x["Volume"])
    latest_x = (latest_x - x_mean) / x_std
    prediction = float(np.dot(np.r_[1, latest_x.iloc[0].to_numpy()], coeffs))
    latest_close = float(clean["Close"].iloc[-1])
    expected_move = (prediction / latest_close) - 1
    return prediction, expected_move


def classify_signal(row: pd.Series, expected_move: float) -> tuple[str, str]:
    above_vwap = row["Close"] >= row["VWAP"]
    rsi = row["RSI"]
    volume_ratio = row["Volume"] / row["Volume_MA"] if row["Volume_MA"] else 1

    if above_vwap and 45 <= rsi <= 68 and expected_move > 0.003:
        return "BUY", "Price above VWAP, RSI constructive, AI tilt positive"
    if (not above_vwap and rsi < 48 and expected_move < -0.003) or rsi > 74:
        return "SELL", "Weak VWAP position or overheated RSI with negative tilt"
    if volume_ratio >= 1.8 and above_vwap:
        return "SMART MONEY", "Unusual volume expansion while holding above VWAP"
    return "WATCH", "No clean high-conviction setup"


def summarize_symbol(symbol: str, period: str, interval: str) -> dict:
    history = fetch_history(symbol, period, interval)
    if history.empty or len(history) < 5:
        return {"Symbol": symbol, "Status": "No data"}

    data = add_indicators(history).dropna()
    if data.empty:
        return {"Symbol": symbol, "Status": "Insufficient data"}

    latest = data.iloc[-1]
    first_close = float(data["Close"].iloc[0])
    last_close = float(latest["Close"])
    prediction, expected_move = predict_next_close(data)
    signal, reason = classify_signal(latest, expected_move)
    volume_ratio = float(latest["Volume"] / latest["Volume_MA"]) if latest["Volume_MA"] else 0
    probability = float(ensemble_probabilities(data).iloc[-1])
    accuracy, samples = ensemble_accuracy(data, 0.58)

    return {
        "Symbol": symbol,
        "Price": last_close,
        "Change %": (last_close / first_close - 1) * 100,
        "AI Next": prediction,
        "AI Move %": expected_move * 100,
        "ML Prob %": probability * 100,
        "ML Accuracy %": accuracy,
        "ML Samples": samples,
        "VWAP": float(latest["VWAP"]),
        "RSI": float(latest["RSI"]),
        "Volume x": volume_ratio,
        "Smart Money": "Yes" if volume_ratio >= 1.8 and last_close >= latest["VWAP"] else "No",
        "Bot Action": bot_action(latest, probability, 0.58),
        "Signal": signal,
        "Reason": reason,
        "Status": "OK",
    }


def run_backtest(
    data: pd.DataFrame,
    threshold: float,
    starting_cash: float,
    position_pct: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    fee_bps: float,
) -> tuple[pd.DataFrame, dict]:
    test = data.dropna(subset=["Close", "VWAP", "RSI", "Volume_MA", "SMA_20", "Momentum_5"]).copy()
    if len(test) < 30:
        return pd.DataFrame(), {"Trades": 0, "Return %": 0.0, "Win Rate %": 0.0, "Max Drawdown %": 0.0}

    test["ML Prob"] = ensemble_probabilities(test)
    cash = starting_cash
    shares = 0.0
    entry_price = 0.0
    trade_returns = []
    equity_rows = []
    fee_rate = fee_bps / 10000

    for timestamp, row in test.iterrows():
        price = float(row["Close"])
        equity = cash + shares * price
        in_position = shares > 0
        exit_signal = (
            in_position
            and (
                price >= entry_price * (1 + take_profit_pct / 100)
                or price <= entry_price * (1 - stop_loss_pct / 100)
                or row["ML Prob"] <= 0.42
                or price < row["VWAP"]
                or row["RSI"] >= 76
            )
        )
        entry_signal = (
            not in_position
            and row["ML Prob"] >= threshold
            and price >= row["VWAP"]
            and 42 <= row["RSI"] <= 72
        )

        action = "HOLD"
        if exit_signal:
            cash += shares * price * (1 - fee_rate)
            trade_returns.append((price / entry_price - 1) * 100)
            shares = 0.0
            entry_price = 0.0
            action = "EXIT"
        elif entry_signal:
            allocation = cash * position_pct / 100
            shares = allocation * (1 - fee_rate) / price
            cash -= allocation
            entry_price = price
            action = "ENTRY"

        equity = cash + shares * price
        equity_rows.append({"Time": timestamp, "Equity": equity, "Action": action, "Price": price, "ML Prob": row["ML Prob"]})

    equity = pd.DataFrame(equity_rows).set_index("Time")
    peak = equity["Equity"].cummax()
    drawdown = ((equity["Equity"] / peak) - 1) * 100
    stats = {
        "Trades": len(trade_returns),
        "Return %": (equity["Equity"].iloc[-1] / starting_cash - 1) * 100,
        "Win Rate %": (np.mean([item > 0 for item in trade_returns]) * 100) if trade_returns else 0.0,
        "Max Drawdown %": float(drawdown.min()) if not drawdown.empty else 0.0,
    }
    return equity, stats


def rebalance_plan(summary: pd.DataFrame, portfolio_value: float, max_weight_pct: float) -> pd.DataFrame:
    data = summary.copy()
    signal_score = data["Signal"].map({"BUY": 1.0, "SMART MONEY": 0.9, "WATCH": 0.35, "SELL": 0.0}).fillna(0.2)
    ml_score = (data["ML Prob %"] / 100).clip(0, 1)
    move_score = sigmoid(data["AI Move %"] / 3)
    raw_score = (0.45 * ml_score + 0.35 * signal_score + 0.20 * move_score).clip(lower=0)
    if raw_score.sum() <= 0:
        data["Target Weight %"] = 0.0
    else:
        data["Target Weight %"] = raw_score / raw_score.sum() * 100
        data["Target Weight %"] = data["Target Weight %"].clip(upper=max_weight_pct)
        data["Target Weight %"] = data["Target Weight %"] / data["Target Weight %"].sum() * 100
    data["Target Value"] = portfolio_value * data["Target Weight %"] / 100
    data["Approx Shares"] = np.floor(data["Target Value"] / data["Price"]).astype(int)
    return data[["Symbol", "Signal", "ML Prob %", "Target Weight %", "Target Value", "Approx Shares"]]


def norm_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)


def black_scholes_greeks(option_type: str, spot: float, strike: float, years: float, rate: float, iv: float) -> dict:
    years = max(years, 1 / 365)
    iv = max(iv, 0.01)
    if spot <= 0 or strike <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0}

    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    gamma = norm_pdf(d1) / (spot * iv * math.sqrt(years))

    if option_type == "call":
        delta = norm_cdf(d1)
        theta = (-(spot * norm_pdf(d1) * iv) / (2 * math.sqrt(years)) - rate * strike * math.exp(-rate * years) * norm_cdf(d2)) / 365
    else:
        delta = norm_cdf(d1) - 1
        theta = (-(spot * norm_pdf(d1) * iv) / (2 * math.sqrt(years)) + rate * strike * math.exp(-rate * years) * norm_cdf(-d2)) / 365

    return {"Delta": delta, "Gamma": gamma, "Theta": theta}


def probability_of_profit(option_type: str, spot: float, strike: float, premium: float, years: float, rate: float, iv: float) -> float:
    years = max(years, 1 / 365)
    iv = max(iv, 0.01)
    premium = max(premium, 0.01)
    if option_type == "call":
        breakeven = strike + premium
        z = (math.log(breakeven / spot) - (rate - 0.5 * iv * iv) * years) / (iv * math.sqrt(years))
        return (1 - norm_cdf(z)) * 100
    breakeven = max(strike - premium, 0.01)
    z = (math.log(breakeven / spot) - (rate - 0.5 * iv * iv) * years) / (iv * math.sqrt(years))
    return norm_cdf(z) * 100


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_expirations(symbol: str) -> list[str]:
    try:
        return list(yf.Ticker(symbol).options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_chain(symbol: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def analyze_option_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiration: str,
    risk_free_rate: float,
    max_spread_pct: float,
    min_volume: int,
    target_delta: float,
    direction: str,
) -> pd.DataFrame:
    expiry_dt = pd.to_datetime(expiration).to_pydatetime().replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    years = max((expiry_dt - now).days / 365, 1 / 365)
    frames = []
    for option_type, chain in [("call", calls), ("put", puts)]:
        if chain.empty:
            continue
        data = chain.copy()
        data["Type"] = option_type.upper()
        data["Mid"] = np.where((data["bid"] > 0) & (data["ask"] > 0), (data["bid"] + data["ask"]) / 2, data["lastPrice"])
        data["Spread %"] = np.where(data["Mid"] > 0, ((data["ask"] - data["bid"]) / data["Mid"]) * 100, 999)
        data["IV %"] = data["impliedVolatility"].fillna(0.01).clip(lower=0.01) * 100
        data["Volume"] = data["volume"].fillna(0)
        data["Open Interest"] = data["openInterest"].fillna(0)
        greek_rows = []
        pops = []
        for row in data.itertuples():
            iv = max(float(row.impliedVolatility or 0.01), 0.01)
            premium = float(row.Mid or row.lastPrice or 0.01)
            strike = float(row.strike)
            greek_rows.append(black_scholes_greeks(option_type, spot, strike, years, risk_free_rate, iv))
            pops.append(probability_of_profit(option_type, spot, strike, premium, years, risk_free_rate, iv))
        greeks = pd.DataFrame(greek_rows, index=data.index)
        data = pd.concat([data, greeks], axis=1)
        data["POP %"] = pops
        data["Breakeven"] = np.where(data["Type"].eq("CALL"), data["strike"] + data["Mid"], data["strike"] - data["Mid"])
        data["Moneyness %"] = ((data["strike"] / spot) - 1) * 100
        frames.append(data)

    if not frames:
        return pd.DataFrame()

    options = pd.concat(frames, ignore_index=True)
    options = options[(options["Mid"] > 0) & (options["Spread %"] <= max_spread_pct) & (options["Volume"] >= min_volume)].copy()
    if options.empty:
        return options

    if direction == "Bullish":
        options = options[options["Type"].eq("CALL")].copy()
    elif direction == "Bearish":
        options = options[options["Type"].eq("PUT")].copy()

    if options.empty:
        return options

    liquidity_score = np.log1p(options["Volume"] + options["Open Interest"]) / np.log1p((options["Volume"] + options["Open Interest"]).max())
    spread_score = (1 - (options["Spread %"] / max_spread_pct)).clip(0, 1)
    delta_score = (1 - (options["Delta"].abs() - target_delta).abs() / max(target_delta, 0.01)).clip(0, 1)
    pop_score = (options["POP %"] / 100).clip(0, 1)
    options["Picker Score"] = (0.34 * pop_score + 0.28 * liquidity_score + 0.23 * delta_score + 0.15 * spread_score) * 100
    options["Bot Action"] = np.where(options["Picker Score"] >= 68, "PAPER BUY", np.where(options["Picker Score"] <= 35, "AVOID", "WATCH"))
    return options.sort_values("Picker Score", ascending=False)


def make_options_scatter(options: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if options.empty:
        return fig
    for option_type, color in [("CALL", "#16a34a"), ("PUT", "#b91c1c")]:
        subset = options[options["Type"].eq(option_type)]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["strike"],
                y=subset["POP %"],
                mode="markers",
                name=option_type,
                marker=dict(size=np.clip(subset["Open Interest"] / max(subset["Open Interest"].max(), 1) * 22 + 7, 7, 28), color=color, opacity=0.72),
                text=subset["contractSymbol"],
                hovertemplate="%{text}<br>Strike %{x}<br>POP %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=25, b=10),
        template="plotly_white",
        paper_bgcolor="#f6f7fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
        xaxis_title="Strike",
        yaxis_title="Probability of Profit %",
    )
    return fig


def alert_message(rows: pd.DataFrame) -> str:
    actionable = rows[rows["Signal"].isin(["BUY", "SELL", "SMART MONEY"])]
    if actionable.empty:
        return ""
    lines = ["Market dashboard alert", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")]
    for _, row in actionable.iterrows():
        lines.append(
            f"{row['Symbol']}: {row['Signal']} | price {row['Price']:.2f} | RSI {row['RSI']:.1f} | "
            f"AI {row['AI Move %']:.2f}% | volume {row['Volume x']:.1f}x"
        )
    return "\n".join(lines)


def post_json(url: str, payload: dict, headers: dict | None = None) -> tuple[bool, str]:
    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300, response.read().decode("utf-8")[:300]
    except Exception as exc:
        return False, str(exc)


def send_telegram(message: str) -> tuple[bool, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first."
    query = urlencode({"chat_id": chat_id, "text": message})
    try:
        with urlopen(f"https://api.telegram.org/bot{token}/sendMessage?{query}", timeout=15) as response:
            return 200 <= response.status < 300, response.read().decode("utf-8")[:300]
    except Exception as exc:
        return False, str(exc)


def send_whatsapp(message: str) -> tuple[bool, str]:
    token = os.getenv("WHATSAPP_TOKEN", "")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    to_number = os.getenv("WHATSAPP_TO", "")
    if not token or not phone_number_id or not to_number:
        return False, "Set WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, and WHATSAPP_TO first."
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }
    return post_json(url, payload, headers={"Authorization": f"Bearer {token}"})


def signal_badge(signal: str) -> str:
    color = {
        "BUY": "#10b981",
        "SELL": "#ef4444",
        "SMART MONEY": "#38bdf8",
        "WATCH": "#f59e0b",
    }.get(signal, "#94a3b8")
    return f"<span style='color:{color};font-weight:800'>{signal}</span>"


def make_price_chart(symbol: str, data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#15803d",
            decreasing_line_color="#b91c1c",
        )
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP", line=dict(color="#2563eb", width=2)))
    fig.update_layout(
        title=f"{symbol} price action",
        height=460,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        paper_bgcolor="#f6f7fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    return fig


def make_rsi_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color="#2563eb")))
    fig.add_hrect(y0=70, y1=100, fillcolor="#fee2e2", opacity=0.9, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#dcfce7", opacity=0.9, line_width=0)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=25, b=10),
        template="plotly_white",
        yaxis_range=[0, 100],
        paper_bgcolor="#f6f7fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    return fig


def make_heatmap(summary: pd.DataFrame) -> go.Figure:
    heat = summary.pivot_table(index="Signal", columns="Symbol", values="Change %", aggfunc="mean").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale=[[0, "#b91c1c"], [0.5, "#f1f5f9"], [1, "#16a34a"]],
            zmid=0,
            text=np.round(heat.values, 2),
            texttemplate="%{text}%",
            hovertemplate="%{x}<br>%{y}<br>%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=25, b=10),
        template="plotly_white",
        paper_bgcolor="#f6f7fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569"),
    )
    return fig


st.markdown("<div class='app-title'>Multi-Stocks AI Trading Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>VWAP, RSI, AI next-bar bias, smart-money volume detection, and Telegram/WhatsApp alerts.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Market setup")
    symbols_text = st.text_area("Symbols", value=", ".join(DEFAULT_SYMBOLS), height=92)
    period = st.selectbox("History", PERIOD_OPTIONS, index=2)
    interval = st.selectbox("Interval", INTERVAL_OPTIONS, index=3)
    selected_symbol = st.text_input("Chart symbol", value=DEFAULT_SYMBOLS[0])
    option_symbol = selected_symbol.strip().upper() or DEFAULT_SYMBOLS[0]
    option_expirations = fetch_option_expirations(option_symbol)
    st.header("Options analyzer")
    option_direction = st.selectbox("Options direction", ["Auto", "Bullish", "Bearish", "Both"], index=0)
    option_expiration = st.selectbox("Expiration", option_expirations, index=0) if option_expirations else ""
    target_delta = st.slider("Target option Delta", min_value=0.10, max_value=0.80, value=0.35, step=0.05)
    max_spread_pct = st.slider("Max option spread %", min_value=5, max_value=100, value=35, step=5)
    min_option_volume = st.number_input("Min option volume", min_value=0, max_value=10000, value=0, step=10)
    risk_free_rate = st.slider("Risk-free rate %", min_value=0.0, max_value=10.0, value=4.5, step=0.1) / 100
    st.header("Bot controls")
    bot_threshold = st.slider("Entry confidence", min_value=0.50, max_value=0.90, value=0.58, step=0.01)
    starting_cash = st.number_input("Backtest cash", min_value=1000, max_value=1000000, value=10000, step=1000)
    position_pct = st.slider("Position size %", min_value=5, max_value=100, value=30, step=5)
    take_profit_pct = st.slider("Take profit %", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
    stop_loss_pct = st.slider("Stop loss %", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
    fee_bps = st.slider("Fee bps", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    portfolio_value = st.number_input("Portfolio value", min_value=1000, max_value=5000000, value=25000, step=1000)
    max_weight_pct = st.slider("Max stock weight %", min_value=5, max_value=60, value=25, step=5)
    send_alerts = st.toggle("Send alerts now", value=False)
    st.caption("Alert credentials are read from environment variables.")

symbols = [item.strip().upper() for item in symbols_text.replace("\n", ",").split(",") if item.strip()]

if not symbols:
    st.warning("Add at least one stock symbol.")
    st.stop()

with st.spinner("Loading market data and computing signals..."):
    rows = [summarize_symbol(symbol, period, interval) for symbol in symbols]
summary = pd.DataFrame(rows)
summary = summary[summary["Status"].eq("OK")]

if summary.empty:
    st.error("No usable market data was returned. Try fewer symbols, a longer period, or a daily interval.")
    st.stop()

summary = summary.sort_values(["Signal", "AI Move %"], ascending=[True, False])
leader = summary.iloc[0]

metric_cols = st.columns(4)
metric_cols[0].metric("Tracked", len(summary))
metric_cols[1].metric("Best AI Move", leader["Symbol"], f"{leader['AI Move %']:.2f}%")
metric_cols[2].metric("Smart Money", int((summary["Smart Money"] == "Yes").sum()))
metric_cols[3].metric("Auto Entries", int((summary["Bot Action"] == "AUTO ENTRY").sum()))

tab_market, tab_bot, tab_options, tab_backtest, tab_portfolio = st.tabs(["Market", "Auto Bot", "Options Bot", "Backtest", "Portfolio AI"])

with tab_market:
    st.subheader("Bloomberg-Style Heatmap")
    st.plotly_chart(make_heatmap(summary), width="stretch")

    st.subheader("Signal Board")
    display = summary.copy()
    display["Signal"] = display["Signal"].map(signal_badge)
    st.write(
        display[
            [
                "Symbol",
                "Price",
                "Change %",
                "AI Next",
                "AI Move %",
                "ML Prob %",
                "ML Accuracy %",
                "VWAP",
                "RSI",
                "Volume x",
                "Smart Money",
                "Signal",
                "Reason",
            ]
        ].to_html(escape=False, index=False, float_format=lambda value: f"{value:,.2f}"),
        unsafe_allow_html=True,
    )

with tab_bot:
    st.subheader("Automated Trading Bot")
    st.caption("Paper-trading automation and broker-ready action instructions. Live order execution needs broker API keys and explicit order permission.")
    bot_board = summary[
        [
            "Symbol",
            "Price",
            "Bot Action",
            "Signal",
            "ML Prob %",
            "ML Accuracy %",
            "AI Move %",
            "RSI",
            "VWAP",
            "Volume x",
        ]
    ].copy()
    st.dataframe(bot_board, width="stretch", hide_index=True)
    actionable_bot = bot_board[bot_board["Bot Action"].isin(["AUTO ENTRY", "AUTO EXIT"])]
    if actionable_bot.empty:
        st.info("No auto entry or exit actions at the current confidence threshold.")
    else:
        st.success("Bot actions are ready for paper execution.")
        st.dataframe(actionable_bot, width="stretch", hide_index=True)

chart_symbol = selected_symbol.strip().upper() or summary.iloc[0]["Symbol"]
history = fetch_history(chart_symbol, period, interval)
with tab_options:
    st.subheader("Full Options Chain Analyzer")
    st.caption("Options bot is paper-trade only here. Live execution needs a broker API connection and explicit order approval.")
    summary_match = summary[summary["Symbol"].eq(chart_symbol)]
    if summary_match.empty:
        st.warning(f"No stock signal data found for {chart_symbol}.")
    elif not option_expiration:
        st.warning(f"No listed option expirations found for {chart_symbol}. Try another symbol such as AAPL, MSFT, NVDA, or TSLA.")
    else:
        spot_price = float(summary_match.iloc[0]["Price"])
        stock_signal = str(summary_match.iloc[0]["Signal"])
        auto_direction = "Bearish" if stock_signal == "SELL" or float(summary_match.iloc[0]["AI Move %"]) < -0.25 else "Bullish"
        resolved_direction = auto_direction if option_direction == "Auto" else option_direction
        with st.spinner(f"Loading {chart_symbol} options chain..."):
            calls, puts = fetch_option_chain(chart_symbol, option_expiration)
            options = analyze_option_chain(
                calls,
                puts,
                spot_price,
                option_expiration,
                float(risk_free_rate),
                float(max_spread_pct),
                int(min_option_volume),
                float(target_delta),
                resolved_direction,
            )
        if options.empty:
            st.warning("No option contracts matched your filters. Try a wider spread limit, lower minimum volume, or another expiration.")
        else:
            best = options.iloc[0]
            opt_cols = st.columns(4)
            opt_cols[0].metric("Underlying", chart_symbol, f"{spot_price:.2f}")
            opt_cols[1].metric("Direction", resolved_direction)
            opt_cols[2].metric("Best Strike", f"{best['Type']} {best['strike']:.2f}")
            opt_cols[3].metric("POP", f"{best['POP %']:.1f}%", f"Score {best['Picker Score']:.1f}")

            st.plotly_chart(make_options_scatter(options), width="stretch")

            st.subheader("Best Strike Auto Picker")
            st.dataframe(
                options[
                    [
                        "contractSymbol",
                        "Type",
                        "strike",
                        "lastPrice",
                        "bid",
                        "ask",
                        "Mid",
                        "Spread %",
                        "impliedVolatility",
                        "IV %",
                        "Volume",
                        "Open Interest",
                        "Delta",
                        "Gamma",
                        "Theta",
                        "POP %",
                        "Breakeven",
                        "Picker Score",
                        "Bot Action",
                    ]
                ].head(25),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Greeks Dashboard")
            greek_cols = st.columns(3)
            greek_cols[0].metric("Delta", f"{best['Delta']:.3f}")
            greek_cols[1].metric("Gamma", f"{best['Gamma']:.4f}")
            greek_cols[2].metric("Theta / day", f"{best['Theta']:.3f}")

            st.subheader("Auto Options Bot")
            if best["Bot Action"] == "PAPER BUY":
                st.success("Paper options bot selected a candidate contract.")
            else:
                st.info("No high-score paper buy candidate at the current filters.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Mode": "PAPER ONLY",
                            "Action": "BUY TO OPEN" if best["Bot Action"] == "PAPER BUY" else "WAIT",
                            "Contract": best["contractSymbol"],
                            "Limit Price": round(float(best["Mid"]), 2),
                            "Take Profit": f"{take_profit_pct:.1f}%",
                            "Stop Loss": f"{stop_loss_pct:.1f}%",
                            "Reason": f"{resolved_direction} setup, POP {best['POP %']:.1f}%, Delta {best['Delta']:.2f}, spread {best['Spread %']:.1f}%",
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
if history.empty:
    st.warning(f"No chart data found for {chart_symbol}.")
else:
    chart_data = add_indicators(history).dropna()
    with tab_market:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(make_price_chart(chart_symbol, chart_data), width="stretch")
        with right:
            st.plotly_chart(make_rsi_chart(chart_data), width="stretch")
            latest_row = summary[summary["Symbol"].eq(chart_symbol)]
            if not latest_row.empty:
                item = latest_row.iloc[0]
                st.markdown(f"### {item['Symbol']} {signal_badge(item['Signal'])}", unsafe_allow_html=True)
                st.write(item["Reason"])
                st.metric("AI next close", f"{item['AI Next']:.2f}", f"{item['AI Move %']:.2f}%")
                st.metric("Ensemble probability", f"{item['ML Prob %']:.1f}%")
                st.metric("Measured accuracy", f"{item['ML Accuracy %']:.1f}%", f"{int(item['ML Samples'])} samples")

    with tab_backtest:
        st.subheader(f"{chart_symbol} Backtesting System")
        equity, stats = run_backtest(
            chart_data,
            bot_threshold,
            float(starting_cash),
            float(position_pct),
            float(take_profit_pct),
            float(stop_loss_pct),
            float(fee_bps),
        )
        stat_cols = st.columns(4)
        stat_cols[0].metric("Return", f"{stats['Return %']:.2f}%")
        stat_cols[1].metric("Trades", stats["Trades"])
        stat_cols[2].metric("Win Rate", f"{stats['Win Rate %']:.1f}%")
        stat_cols[3].metric("Max Drawdown", f"{stats['Max Drawdown %']:.2f}%")
        if equity.empty:
            st.warning("Not enough clean historical data to backtest this symbol and interval.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity.index, y=equity["Equity"], mode="lines", name="Equity", line=dict(color="#2563eb")))
            entries = equity[equity["Action"].eq("ENTRY")]
            exits = equity[equity["Action"].eq("EXIT")]
            fig.add_trace(go.Scatter(x=entries.index, y=entries["Equity"], mode="markers", name="Entry", marker=dict(color="#16a34a", size=9)))
            fig.add_trace(go.Scatter(x=exits.index, y=exits["Equity"], mode="markers", name="Exit", marker=dict(color="#b91c1c", size=9)))
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=25, b=10),
                template="plotly_white",
                paper_bgcolor="#f6f7fb",
                plot_bgcolor="#ffffff",
                font=dict(color="#334155"),
            )
            st.plotly_chart(fig, width="stretch")

with tab_portfolio:
    st.subheader("Portfolio Rebalancing AI")
    st.caption("Long-only target allocation using signal quality, ensemble probability, and AI move estimate.")
    plan = rebalance_plan(summary, float(portfolio_value), float(max_weight_pct))
    st.dataframe(plan, width="stretch", hide_index=True)

message = alert_message(summary)
if send_alerts:
    if not message:
        st.info("No actionable BUY, SELL, or SMART MONEY alerts right now.")
    else:
        telegram_ok, telegram_result = send_telegram(message)
        whatsapp_ok, whatsapp_result = send_whatsapp(message)
        st.write("Telegram:", "sent" if telegram_ok else telegram_result)
        st.write("WhatsApp:", "sent" if whatsapp_ok else whatsapp_result)

with st.expander("Alert configuration"):
    st.code(
        """
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
$env:WHATSAPP_TOKEN="your_meta_cloud_api_token"
$env:WHATSAPP_PHONE_NUMBER_ID="your_phone_number_id"
$env:WHATSAPP_TO="recipient_number_with_country_code"
streamlit run stock_dashboard_app.py
        """.strip(),
        language="powershell",
    )
