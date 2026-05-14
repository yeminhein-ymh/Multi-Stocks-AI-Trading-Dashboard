import json
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
    .main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #243044;
        border-radius: 8px;
        padding: 12px 14px;
        color: white;
    }
    [data-testid="stMetric"] label { color: #cbd5e1 !important; }
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
    return data


def predict_next_close(df: pd.DataFrame) -> tuple[float, float]:
    clean = df[["Close", "VWAP", "RSI", "Volume"]].dropna().tail(80)
    if len(clean) < 20:
        latest = float(df["Close"].dropna().iloc[-1])
        return latest, 0.0

    y = clean["Close"].shift(-1).dropna()
    x = clean.loc[y.index, ["Close", "VWAP", "RSI", "Volume"]].copy()
    x["Volume"] = np.log1p(x["Volume"])
    x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coeffs = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]

    latest_x = clean[["Close", "VWAP", "RSI", "Volume"]].tail(1).copy()
    latest_x["Volume"] = np.log1p(latest_x["Volume"])
    latest_x = (latest_x - x.mean()) / x.std(ddof=0).replace(0, 1)
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

    return {
        "Symbol": symbol,
        "Price": last_close,
        "Change %": (last_close / first_close - 1) * 100,
        "AI Next": prediction,
        "AI Move %": expected_move * 100,
        "VWAP": float(latest["VWAP"]),
        "RSI": float(latest["RSI"]),
        "Volume x": volume_ratio,
        "Smart Money": "Yes" if volume_ratio >= 1.8 and last_close >= latest["VWAP"] else "No",
        "Signal": signal,
        "Reason": reason,
        "Status": "OK",
    }


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
        )
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP", line=dict(color="#f59e0b", width=2)))
    fig.update_layout(
        title=f"{symbol} price action",
        height=460,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    return fig


def make_rsi_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color="#22c55e")))
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.14, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.14, line_width=0)
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=25, b=10), template="plotly_dark", yaxis_range=[0, 100])
    return fig


def make_heatmap(summary: pd.DataFrame) -> go.Figure:
    heat = summary.pivot_table(index="Signal", columns="Symbol", values="Change %", aggfunc="mean").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale=[[0, "#991b1b"], [0.5, "#111827"], [1, "#16a34a"]],
            zmid=0,
            text=np.round(heat.values, 2),
            texttemplate="%{text}%",
            hovertemplate="%{x}<br>%{y}<br>%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(height=310, margin=dict(l=10, r=10, t=25, b=10), template="plotly_dark")
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
metric_cols[3].metric("Buy Signals", int((summary["Signal"] == "BUY").sum()))

st.subheader("Bloomberg-Style Heatmap")
st.plotly_chart(make_heatmap(summary), width="stretch")

st.subheader("Signal Board")
display = summary.copy()
display["Signal"] = display["Signal"].map(signal_badge)
st.write(
    display[
        ["Symbol", "Price", "Change %", "AI Next", "AI Move %", "VWAP", "RSI", "Volume x", "Smart Money", "Signal", "Reason"]
    ].to_html(escape=False, index=False, float_format=lambda value: f"{value:,.2f}"),
    unsafe_allow_html=True,
)

chart_symbol = selected_symbol.strip().upper() or summary.iloc[0]["Symbol"]
history = fetch_history(chart_symbol, period, interval)
if history.empty:
    st.warning(f"No chart data found for {chart_symbol}.")
else:
    chart_data = add_indicators(history).dropna()
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
            st.metric("Volume expansion", f"{item['Volume x']:.2f}x")

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
