# Multi-Stocks AI Trading Dashboard

Streamlit dashboard with multi-stock monitoring, AI-style next-close bias, VWAP and RSI signals, Bloomberg-style heatmap, smart-money volume detection, Telegram alerts, WhatsApp Cloud API alerts, and responsive layout.

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
