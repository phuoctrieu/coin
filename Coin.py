import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from arch import arch_model
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import statsmodels.api as sm
import pytz
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
import requests_cache

# Thiết lập cache cho requests
requests_cache.install_cache('binance_cache', backend='sqlite', expire_after=300)

# URL API Binance
API_URL = "https://api.binance.com/api/v3/klines"

# Đặt múi giờ Việt Nam
vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Hàm để lấy dữ liệu coin
@st.cache_data
def get_coin_data(symbol, interval, start_time):
    end_time = int((datetime.now() - timedelta(minutes=1)).timestamp() * 1000)
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    df["Close"] = pd.to_numeric(df["Close"])

    # Chuyển đổi thời gian từ UTC sang múi giờ Việt Nam
    df["Open time"] = df["Open time"].dt.tz_localize('UTC').dt.tz_convert(vietnam_tz)
    df["Close time"] = df["Close time"].dt.tz_localize('UTC').dt.tz_convert(vietnam_tz)

    return df

# Hàm để lấy danh sách các đồng coin
@st.cache_data
def get_available_symbols():
    try:
        headers = {
            "Accept": "application/json",
            "X-MBX-SBE": "1:0"
        }
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo", headers=headers)
        response.raise_for_status()
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
        return symbols
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching available symbols: {e}")
        return []

# Hàm dự đoán giá sử dụng GARCH model
@st.cache_data
def predict_price_garch(df, horizon, p=1, q=1):  # Thêm tham số p, q
    returns = df["Close"].pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=p, q=q)
    model_fitted = model.fit(disp="off")
    forecast = model_fitted.forecast(horizon=horizon)
    predicted_volatility = forecast.variance.values[-1, :]
    last_price = df["Close"].iloc[-1]
    predicted_return = returns.mean()
    predicted_prices = [last_price * (1 + predicted_return) for _ in range(horizon)]
    return predicted_prices, predicted_volatility

# Hàm dự đoán giá sử dụng Seasonal ARIMA
@st.cache_data
def predict_price_sarima(df, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    close_prices = df["Close"]
    model = sm.tsa.SARIMAX(close_prices, order=order, seasonal_order=seasonal_order)
    model_fitted = model.fit(disp=False)
    forecast = model_fitted.get_forecast(steps=horizon)
    predicted_prices = forecast.predicted_mean.values
    return predicted_prices

# Hàm dự đoán giá sử dụng GRU
@st.cache_data
def predict_price_gru(df, horizon, n_steps=10, epochs=100, batch_size=32):  # Thêm epochs, batch_size
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(n_steps, len(scaled_data) - horizon + 1):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i:i + horizon, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(tf.keras.layers.GRU(50))
    model.add(tf.keras.layers.Dense(horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    last_data = scaled_data[-n_steps:].reshape(1, n_steps, 1)
    predicted_prices = model.predict(last_data)
    predicted_prices = scaler.inverse_transform(predicted_prices).flatten()
    return predicted_prices

# Hàm dự đoán giá sử dụng Moving Averages
def predict_price_moving_average(df, short_window=5, long_window=20):
    df['Short MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long MA'] = df['Close'].rolling(window=long_window).mean()
    last_short_ma = df['Short MA'].iloc[-1]
    last_long_ma = df['Long MA'].iloc[-1]
    prediction = "Mua" if last_short_ma > last_long_ma else "Bán"
    return last_short_ma, last_long_ma, prediction

# Hàm dự đoán giá sử dụng Exponential Moving Average (EMA)
def predict_price_ema(df, window=20):
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    last_ema = df['EMA'].iloc[-1]
    prediction = "Mua" if df['Close'].iloc[-1] > last_ema else "Bán"
    return last_ema, prediction

# Hàm dự đoán giá sử dụng Price Action
def predict_price_action(df, window=10):  # Thêm tham số window
    close_prices = df['Close'].values
    support = np.min(close_prices[-window:])
    resistance = np.max(close_prices[-window:])
    last_price = close_prices[-1]
    if last_price < support:
        prediction = "Bán"
    elif last_price > resistance:
        prediction = "Mua"
    else:
        prediction = "Giữ"
    return support, resistance, prediction

# Hàm để tính toán điểm mua vào và điểm chốt lời
def calculate_entry_and_target_points(predicted_prices, last_price, risk_reward_ratio=2):
    entry_points = []
    target_points = []
    for price in predicted_prices:
        entry_points.append(last_price)  # Điểm mua vào là giá hiện tại
        target_points.append(price + (price - last_price) * risk_reward_ratio)  # Điểm chốt lời
    return entry_points, target_points

# Hàm tính toán Average True Range (ATR)
def calculate_atr(df, period=14):
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    atr = df['TR'].rolling(window=period).mean()
    
    return atr

# Hàm tính toán Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['MA'] + (df['Close'].rolling(window=window).std() * num_std_dev)
    df['Lower Band'] = df['MA'] - (df['Close'].rolling(window=window).std() * num_std_dev)
    
    return df[['MA', 'Upper Band', 'Lower Band']]

# Giao diện Streamlit
st.title("Dự đoán giá Coin")

# Lấy danh sách các đồng coin có sẵn
available_symbols = get_available_symbols()
if not available_symbols:
    st.error("Không thể lấy danh sách các đồng coin.")
else:
    # Chọn cặp coin, khoảng thời gian, thời gian bắt đầu và kết thúc
    symbol = st.sidebar.selectbox("Chọn cặp coin", available_symbols)
    interval = st.sidebar.selectbox("Chọn khoảng thời gian", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2)
    now = datetime.now()
    start_time = st.sidebar.date_input("Chọn ngày bắt đầu", now - timedelta(days=1))
    end_time = st.sidebar.date_input("Chọn ngày kết thúc", now)
    start_datetime = datetime.combine(start_time, datetime.min.time())
    end_datetime = datetime.combine(end_time, datetime.max.time())
    start_timestamp = int(start_datetime.timestamp() * 1000)
    end_timestamp = int(end_datetime.timestamp() * 1000)

    # Lấy dữ liệu
    df = get_coin_data(symbol, interval, start_timestamp)

    # Hiển thị dữ liệu
    st.write("Dữ liệu giá coin:")
    st.dataframe(df, use_container_width=True)

    # Chọn số bước dự đoán
    horizon = st.sidebar.slider("Chọn số bước dự đoán", 1, 10, 1)

    # Dự đoán giá
    predicted_prices_garch, predicted_volatility = predict_price_garch(df, horizon)
    predicted_prices_sarima = predict_price_sarima(df, horizon)
    predicted_prices_gru = predict_price_gru(df, horizon)

    # Tính toán điểm mua vào và điểm chốt lời
    last_price = df["Close"].iloc[-1]
    entry_points_garch, target_points_garch = calculate_entry_and_target_points(predicted_prices_garch, last_price)
    entry_points_sarima, target_points_sarima = calculate_entry_and_target_points(predicted_prices_sarima, last_price)
    entry_points_gru, target_points_gru = calculate_entry_and_target_points(predicted_prices_gru, last_price)

    

    # Dự đoán giá
    last_short_ma, last_long_ma, moving_average_prediction = predict_price_moving_average(df)
    last_ema, ema_prediction = predict_price_ema(df)
    support, resistance, price_action_prediction = predict_price_action(df)

    # Hiển thị kết quả dự đoán
    predictions_df = pd.DataFrame({
        "Bước": [f"Bước {i+1}" for i in range(horizon)],
        "Dự đoán GARCH": predicted_prices_garch,
        "Dự đoán SARIMA": predicted_prices_sarima,
        "Dự đoán GRU": predicted_prices_gru,
        "Short MA": last_short_ma,
        "Long MA": last_long_ma,
        "EMA": last_ema,
        "Mức hỗ trợ": support,
        "Mức kháng cự": resistance,
    })
    st.write("Dự đoán giá coin tiếp theo cho các bước:")
    st.dataframe(predictions_df, use_container_width=True)

    # Hiển thị kết luận
    conclusions = []
    for i in range(horizon):
        garch_conclusion = "Mua" if predicted_prices_garch[i] > df["Close"].iloc[-1] else "Bán"
        sarima_conclusion = "Mua" if predicted_prices_sarima[i] > df["Close"].iloc[-1] else "Bán"
        gru_conclusion = "Mua" if predicted_prices_gru[i] > df["Close"].iloc[-1] else "Bán"
        conclusions.append({
            "Bước": f"Bước {i+1}",
            "Kết luận GARCH": garch_conclusion,
            "Kết luận SARIMA": sarima_conclusion,
            "Kết luận GRU": gru_conclusion,
            "Kết luận MA": moving_average_prediction,
            "Kết luận EMA": ema_prediction,
            "Kết luận Price Action": price_action_prediction
        })
    conclusions_df = pd.DataFrame(conclusions)
    st.write("Kết luận cho từng bước:")
    st.dataframe(conclusions_df, use_container_width=True)

    
    # Hiển thị bảng điểm mua vào và điểm bán
    entry_target_df = pd.DataFrame({
        "Bước": [f"Bước {i+1}" for i in range(horizon)],
        "Điểm mua GARCH": entry_points_garch,
        "Điểm bán GARCH": target_points_garch,  # Thêm cột "Điểm bán GARCH"
        "Điểm mua SARIMA": entry_points_sarima,
        "Điểm bán SARIMA": target_points_sarima,  # Thêm cột "Điểm bán SARIMA"
        "Điểm mua GRU": entry_points_gru,
        "Điểm bán GRU": target_points_gru,  # Thêm cột "Điểm bán GRU"
    })

    st.write("Bảng điểm mua vào và điểm bán:")
    st.dataframe(entry_target_df, use_container_width=True)

    # Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df["Close"],
                             mode='lines', name='Giá đóng cửa', line=dict(color='blue')))
    for i in range(horizon):
        # Tính toán thời gian dự đoán dựa trên khoảng thời gian của dữ liệu
        predicted_time = df["Open time"].iloc[-1] + pd.to_timedelta((i + 1) * int(interval[:-1]), unit=interval[-1])
        
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), predicted_time],
                                 y=[df["Close"].iloc[-1], predicted_prices_garch[i]],
                                 mode='lines+text', name=f'Dự đoán giá GARCH Bước {i+1}',
                                 line=dict(color='red', dash='dash'),
                                 text=['', f'{predicted_prices_garch[i]:.2f}'],
                                 textposition='top right'))
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), predicted_time],
                                 y=[df["Close"].iloc[-1], predicted_prices_sarima[i]],
                                 mode='lines+text', name=f'Dự đoán giá SARIMA Bước {i+1}',
                                 line=dict(color='green', dash='dash'),
                                 text=['', f'{predicted_prices_sarima[i]:.2f}'],
                                 textposition='top right'))
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), predicted_time],
                                 y=[df["Close"].iloc[-1], predicted_prices_gru[i]],
                                 mode='lines+text', name=f'Dự đoán giá GRU Bước {i+1}',
                                 line=dict(color='orange', dash='dash'),
                                 text=['', f'{predicted_prices_gru[i]:.2f}'],
                                 textposition='top right'))
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['Short MA'],
                             mode='lines', name='Short MA', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['Long MA'],
                             mode='lines', name='Long MA', line=dict(color='pink')))
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['EMA'],
                             mode='lines', name='EMA', line=dict(color='cyan')))
    fig.update_layout(
        title=f"Biểu đồ giá {symbol}",
        xaxis_title="Thời gian",
        yaxis_title="Giá (USD)",
        template="plotly_dark",
        width=1200,
        height=600
    )
    st.plotly_chart(fig)

    # Kiểm tra kiểu dữ liệu của các cột
    print(df.dtypes)

    # Chuyển đổi các cột "High", "Low", "Close" thành kiểu số
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Kiểm tra lại kiểu dữ liệu sau khi chuyển đổi
    print(df.dtypes)

    # Tính toán ATR
    df['ATR'] = calculate_atr(df)

    # Tính toán Bollinger Bands
    bollinger_bands = calculate_bollinger_bands(df)

    # Hiển thị mô tả về ATR
    st.write("### Chỉ số kỹ thuật Average True Range (ATR)")
    st.write("""
    Average True Range (ATR) là một chỉ số đo lờng độ biến động của giá. 
    ATR không chỉ đo lường sự thay đổi giá giữa các phiên mà còn tính đến các khoảng cách giữa giá cao nhất và thấp nhất trong một khoảng thời gian nhất định.
    - **Cách tính ATR**:
      - True Range (TR) là giá trị lớn nhất trong ba giá trị:
        1. Khoảng cách giữa giá cao nhất và giá thấp nhất trong phiên hiện tại.
        2. Khoảng cách giữa giá cao nhất trong phiên hiện tại và giá đóng cửa của phiên trước.
        3. Khoảng cách giữa giá thấp nhất trong phiên hiện tại và giá đóng cửa của phiên trước.
      - ATR là trung bình của TR trong một khoảng thời gian nhất định (thường là 14 phiên).
    """)
    st.write("""
    - **Ý nghĩa**: 
      - Biến động cao có thể là tín hiệu để điều chỉnh mức dừng lỗ hoặc chốt lời.
      - Mua khi ATR tăng trong xu hướng tăng; Bán khi ATR tăng trong xu hướng giảm.""")

    # Hiển thị ATR
    st.write("Chỉ số kỹ thuật ATR:")
    st.dataframe(df[['Open time', 'ATR']], use_container_width=True)

    # Hiển thị mô tả về Bollinger Bands
    st.write("### Chỉ số kỹ thuật Bollinger Bands")
    st.write("""
    Bollinger Bands là một chỉ số kỹ thuật bao gồm một đường trung bình động (MA) và hai đường biên (upper band và lower band) được tính toán dựa trên độ lệch chuẩn của giá.
    - **Cách tính Bollinger Bands**:
      - Đường trung bình động (MA) là trung bình của giá đóng cửa trong một khoảng thời gian nhất định (thường là 20 phiên).
      - Upper Band được tính bằng MA cộng với độ lệch chuẩn của giá đóng cửa nhân với một hệ số (thường là 2).
      - Lower Band được tính bằng MA trừ đi độ lệch chuẩn của giá đóng cửa nhân với một hệ số (thường là 2).
    """)


    # Hiển thị Bollinger Bands
    st.write("Chỉ số kỹ thuật Bollinger Bands:")
    st.dataframe(bollinger_bands, use_container_width=True)
    st.write("""
        - **Ý nghĩa Average (MA)**: 
      - Xác định xu hướng hiện tại của thị trường.
      - Mua khi giá cắt lên trên MA; Bán khi giá cắt xuống dưới MA.

    - **Ý nghĩa Upper Band và Lower Band (Bollinger Bands)**: 
      - Upper Band cho thấy tài sản có thể bị mua quá mức; Lower Band cho thy tài sản có thể bị bán quá mức.
      - Mua khi giá chạm vào Lower Band và có dấu hiệu đảo chiều; Bán khi giá chạm vào Upper Band và có dấu hiệu đảo chiều.
    """)
    
