# Multi-Stocks AI Trading Dashboard

Streamlit dashboard with multi-stock monitoring, AI-style next-close bias, VWAP and RSI signals, Bloomberg-style heatmap, smart-money volume detection, Telegram alerts, WhatsApp Cloud API alerts, responsive layout, paper-trading bot controls, backtesting, portfolio rebalancing, and options-chain analysis.

## Options Features

- Full calls/puts chain analyzer from Yahoo Finance data
- Best strike auto picker
- Delta, Gamma, and Theta dashboard
- Probability of profit estimate
- Paper-only options bot ticket with suggested action, limit price, take profit, and stop loss

Live stock or options order execution is intentionally not enabled by default. Connect a broker API only after reviewing order permissions, risk controls, and local regulations.

## Run

```powershell
streamlit run stock_dashboard_app.py
```

If Streamlit is not on your PATH, run it with Python:

```powershell
& "C:\Users\Ye Min Hein\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run stock_dashboard_app.py
```

## Alert Environment Variables

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
$env:WHATSAPP_TOKEN="your_meta_cloud_api_token"
$env:WHATSAPP_PHONE_NUMBER_ID="your_phone_number_id"
$env:WHATSAPP_TO="recipient_number_with_country_code"
```

The predictions are lightweight quantitative estimates based on recent price, VWAP, RSI, and volume. They are useful for screening, not financial advice.
