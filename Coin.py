import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from arch import arch_model  # GARCH model
import numpy as np
from prophet import Prophet  # Prophet model
import plotly.graph_objects as go  # Thư viện plotly
import statsmodels.api as sm
import pytz
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# URL API Binance
API_URL = "https://api.binance.com/api/v3/klines"

# Đặt múi giờ Việt Nam
vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Hàm để lấy dữ liệu coin
@st.cache_data
def get_coin_data(symbol, interval, start_time):
    # Sử dụng thời gian hiện tại làm end_time
    end_time = int((datetime.now() - timedelta(minutes=1)).timestamp() * 1000)  # Thời gian hiện tại trừ 1 phút
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time  # Sử dụng end_time hiện tại
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    df["Close"] = pd.to_numeric(df["Close"])  # Ensure Close is numeric

    # Chuyển đổi thời gian từ UTC sang múi giờ Việt Nam
    df["Open time"] = df["Open time"].dt.tz_localize('UTC').dt.tz_convert(vietnam_tz)
    df["Close time"] = df["Close time"].dt.tz_localize('UTC').dt.tz_convert(vietnam_tz)

    return df

# Hàm để lấy danh sách các đồng coin
def get_available_symbols():
    try:
        response = requests.get("https://data.binance.com/api/v3")
        response.raise_for_status()
        symbols = [s['symbol'] for s in response.json()['symbols'] if s['status'] == 'TRADING']
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching symbols: {e}")
        return []
    return symbols

# Hàm dự đoán giá sử dụng GARCH model
def predict_price_garch(df, horizon):
    # Lấy dữ liệu giá đóng cửa
    returns = df["Close"].pct_change().dropna()  # Tính tỷ lệ thay đổi giá (log return)

    # Áp dụng mô hình GARCH(1, 1)
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fitted = model.fit(disp="off")

    # Dự đoán biến động giá trong nhiều bước
    forecast = model_fitted.forecast(horizon=horizon)
    predicted_volatility = forecast.variance.values[-1, :]  # Dự báo độ biến động cho các bước

    # Dự báo giá theo xu hướng hiện tại
    last_price = df["Close"].iloc[-1]
    predicted_return = returns.mean()  # Trung bình thay đổi giá

    # Dự đoán giá cho các bước tới
    predicted_prices = [last_price * (1 + predicted_return) for _ in range(horizon)]
    
    return predicted_prices, predicted_volatility

# Hàm dự đoán giá sử dụng Seasonal ARIMA
def predict_price_arima(df, horizon):
    # Lấy dữ liệu giá đóng cửa
    close_prices = df["Close"]

    # Khởi tạo và huấn luyện mô hình Seasonal ARIMA
    model = sm.tsa.SARIMAX(close_prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))  # Điều chỉnh tham số theo nhu cầu
    model_fitted = model.fit(disp=False)

    # Dự đoán giá cho các bước tiếp theo
    forecast = model_fitted.get_forecast(steps=horizon)
    predicted_prices = forecast.predicted_mean.values
    
    return predicted_prices

# Hàm dự đoán giá sử dụng Moving Averages
def predict_price_moving_average(df, short_window=5, long_window=20):
    # Tính toán Moving Averages
    df['Short MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long MA'] = df['Close'].rolling(window=long_window).mean()

    # Lấy giá trị cuối cùng của Short MA và Long MA
    last_short_ma = df['Short MA'].iloc[-1]
    last_long_ma = df['Long MA'].iloc[-1]

    # Dự đoán dựa trên Moving Averages
    if last_short_ma > last_long_ma:
        prediction = "Mua"
    else:
        prediction = "Bán"

    return last_short_ma, last_long_ma, prediction

# Hàm dự đoán giá sử dụng Exponential Moving Average (EMA)
def predict_price_ema(df, window=20):
    # Tính toán EMA
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()

    # Lấy giá trị cuối cùng của EMA
    last_ema = df['EMA'].iloc[-1]

    # Dự đoán dựa trên EMA
    if df['Close'].iloc[-1] > last_ema:
        prediction = "Mua"
    else:
        prediction = "Bán"

    return last_ema, prediction

# Hàm dự đoán giá sử dụng Seasonal ARIMA
def predict_price_sarima(df, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    # Lấy dữ liệu giá đóng cửa
    close_prices = df["Close"]

    # Khởi tạo và huấn luyện mô hình Seasonal ARIMA
    model = sm.tsa.SARIMAX(close_prices, order=order, seasonal_order=seasonal_order)
    model_fitted = model.fit(disp=False)

    # Dự đoán giá cho các bước tiếp theo
    forecast = model_fitted.get_forecast(steps=horizon)
    predicted_prices = forecast.predicted_mean.values
    
    return predicted_prices

# Hàm dự đoán giá sử dụng GRU
def predict_price_gru(df, horizon, n_steps=10):
    # Chọn cột giá đóng cửa
    data = df['Close'].values
    data = data.reshape(-1, 1)

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Tạo dữ liệu cho mô hình
    X, y = [], []
    for i in range(n_steps, len(scaled_data) - horizon + 1):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i:i + horizon, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Định dạng cho GRU

    # Xây dựng mô hình GRU
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(tf.keras.layers.GRU(50))
    model.add(tf.keras.layers.Dense(horizon))  # Đầu ra cho số bước dự đoán

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện mô hình
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Dự đoán
    last_data = scaled_data[-n_steps:].reshape(1, n_steps, 1)
    predicted_prices = model.predict(last_data)
    predicted_prices = scaler.inverse_transform(predicted_prices)  # Chuyển đổi về giá thực

    return predicted_prices.flatten()

# Hàm dự đoán giá sử dụng Price Action
def predict_price_action(df):
    # Lấy giá đóng cửa
    close_prices = df['Close'].values

    # Tính toán các mức hỗ trợ và kháng cự
    support = np.min(close_prices[-10:])  # Mức hỗ trợ là giá thấp nhất trong 10 phiên gần nhất
    resistance = np.max(close_prices[-10:])  # Mức kháng cự là giá cao nhất trong 10 phiên gần nhất

    # Dự đoán dựa trên hành động giá
    last_price = close_prices[-1]
    if last_price < support:
        prediction = "Bán"  # Giá dưới mức hỗ trợ
    elif last_price > resistance:
        prediction = "Mua"  # Giá trên mức kháng cự
    else:
        prediction = "Giữ"  # Giá nằm trong khoảng hỗ trợ và kháng cự

    return support, resistance, prediction

# Giao diện Streamlit
st.title("Dự đoán giá Coin với GARCH, ARIMA, Moving Averages, EMA, Seasonal ARIMA, GRU và Price Action")

# Lấy danh sách các đồng coin có sẵn
available_symbols = get_available_symbols()
if not available_symbols:
    st.error("Không thể lấy danh sách các đồng coin.")
else:
    # Chọn cặp coin từ sidebar
    symbol = st.sidebar.selectbox("Chọn cặp coin", available_symbols)

    # Chọn khoảng thời gian (interval) từ sidebar
    interval = st.sidebar.selectbox("Chọn khoảng thời gian", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])

    # Chọn thời gian bắt đầu và kết thúc từ sidebar
    now = datetime.now()
    start_time = st.sidebar.date_input("Chọn ngày bắt đầu", now - timedelta(days=1))
    end_time = st.sidebar.date_input("Chọn ngày kết thúc", now)

    # Chuyển đổi thời gian thành timestamp (ms)
    start_datetime = datetime.combine(start_time, datetime.min.time())
    end_datetime = datetime.combine(end_time, datetime.max.time())

    start_timestamp = int(start_datetime.timestamp() * 1000)
    end_timestamp = int(end_datetime.timestamp() * 1000)

    # Lấy dữ liệu
    df = get_coin_data(symbol, interval, start_timestamp)

    # Chuyển đổi thời gian từ UTC sang múi giờ Việt Nam
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms").dt.tz_convert('UTC').dt.tz_convert(vietnam_tz)
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms").dt.tz_convert('UTC').dt.tz_convert(vietnam_tz)

    # Hiển thị dữ liệu
    st.write("Dữ liệu giá coin:")
    st.dataframe(df, use_container_width=True)  # Sử dụng toàn bộ chiều rộng

    # Thêm slider để điều chỉnh số bước dự đoán (horizon)
    horizon = st.sidebar.slider("Chọn số bước dự đoán", min_value=1, max_value=10, value=1)

    # Dự đoán giá sử dụng GARCH
    predicted_prices_garch, predicted_volatility = predict_price_garch(df, horizon)

    # Dự đoán giá sử dụng Seasonal ARIMA
    predicted_prices_arima = predict_price_arima(df, horizon)

    # Dự đoán giá sử dụng Moving Averages
    last_short_ma, last_long_ma, moving_average_prediction = predict_price_moving_average(df)

    # Dự đoán giá sử dụng EMA
    last_ema, ema_prediction = predict_price_ema(df)

    # Dự đoán giá sử dụng Seasonal ARIMA
    predicted_prices_sarima = predict_price_sarima(df, horizon)

    # Dự đoán giá sử dụng GRU
    predicted_prices_gru = predict_price_gru(df, horizon)

    # Dự đoán giá sử dụng Price Action
    support, resistance, price_action_prediction = predict_price_action(df)

    # Tạo DataFrame để hiển thị kết quả dự đoán
    predictions_df = pd.DataFrame({
        "Bước": [f"Bước {i+1}" for i in range(horizon)],
        "Dự đoán GARCH": predicted_prices_garch,
        "Dự đoán ARIMA": predicted_prices_arima,
        "Dự đoán SARIMA": predicted_prices_sarima,  # Thêm cột cho SARIMA
        "Dự đoán GRU": predicted_prices_gru,  # Thêm cột cho GRU
        "Short MA": last_short_ma,
        "Long MA": last_long_ma,
        #"Kết luận MA": moving_average_prediction,
        "EMA": last_ema,
        #"Kết luận EMA": ema_prediction,
        "Mức hỗ trợ": support,
        "Mức kháng cự": resistance,
        #"Kết luận Price Action": price_action_prediction
    })

    # Hiển thị kết quả dự đoán
    st.write("Dự đoán giá coin tiếp theo cho các bước:")
    st.dataframe(predictions_df, use_container_width=True)  # Sử dụng toàn bộ chiều rộng

    # Tạo DataFrame để lưu kết luận
    conclusions = []
    for i in range(horizon):
        garch_conclusion = "Mua" if predicted_prices_garch[i] > df["Close"].iloc[-1] else "Bán"
        arima_conclusion = "Mua" if predicted_prices_arima[i] > df["Close"].iloc[-1] else "Bán"
        sarima_conclusion = "Mua" if predicted_prices_sarima[i] > df["Close"].iloc[-1] else "Bán"
        gru_conclusion = "Mua" if predicted_prices_gru[i] > df["Close"].iloc[-1] else "Bán"

        conclusions.append({
            "Bước": f"Bước {i+1}",
            "Kết luận GARCH": garch_conclusion,
            "Kết luận ARIMA": arima_conclusion,
            "Kết luận SARIMA": sarima_conclusion,  # Thêm cột cho SARIMA
            "Kết luận GRU": gru_conclusion,  # Thêm cột cho GRU
            "Kết luận MA": moving_average_prediction,  # Thêm cột cho MA
            "Kết luận EMA": ema_prediction,  # Thêm cột cho EMA
            "Kết luận Price Action": price_action_prediction  # Thêm cột cho Price Action
        })

    conclusions_df = pd.DataFrame(conclusions)

    # Hiển thị kết luận
    st.write("Kết luận cho từng bước:")
    st.dataframe(conclusions_df, use_container_width=True)  # Sử dụng toàn bộ chiều rộng

    # Vẽ biểu đồ với Plotly
    fig = go.Figure()

    # Vẽ biểu đồ giá đóng cửa
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df["Close"],
                             mode='lines', name='Giá đóng cửa', line=dict(color='blue')))

    # Vẽ dự đoán giá từ GARCH cho từng bước
    for i in range(horizon):
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), 
                                     df["Open time"].iloc[-1].tz_localize(None) + timedelta(hours=i+1)],
                                 y=[df["Close"].iloc[-1], predicted_prices_garch[i]],
                                 mode='lines+text', name=f'Dự đoán giá GARCH Bước {i+1}',
                                 line=dict(color='red', dash='dash'),
                                 text=['', f'{predicted_prices_garch[i]:.2f}'],
                                 textposition='top right'))

    # Vẽ dự đoán giá từ ARIMA cho từng bước
    for i in range(horizon):
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), 
                                     df["Open time"].iloc[-1].tz_localize(None) + timedelta(hours=i+1)],
                                 y=[df["Close"].iloc[-1], predicted_prices_arima[i]],
                                 mode='lines+text', name=f'Dự đoán giá ARIMA Bước {i+1}',
                                 line=dict(color='orange', dash='dash'),
                                 text=['', f'{predicted_prices_arima[i]:.2f}'],
                                 textposition='top right'))

    # Vẽ dự đoán giá từ SARIMA cho từng bước
    for i in range(horizon):
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), 
                                     df["Open time"].iloc[-1].tz_localize(None) + timedelta(hours=i+1)],
                                 y=[df["Close"].iloc[-1], predicted_prices_sarima[i]],
                                 mode='lines+text', name=f'Dự đoán giá SARIMA Bước {i+1}',
                                 line=dict(color='purple', dash='dash'),
                                 text=['', f'{predicted_prices_sarima[i]:.2f}'],
                                 textposition='top right'))

    # Vẽ dự đoán giá từ GRU cho từng bước
    for i in range(horizon):
        fig.add_trace(go.Scatter(x=[df["Open time"].iloc[-1].tz_localize(None), 
                                     df["Open time"].iloc[-1].tz_localize(None) + timedelta(hours=i+1)],
                                 y=[df["Close"].iloc[-1], predicted_prices_gru[i]],
                                 mode='lines+text', name=f'Dự đoán giá GRU Bước {i+1}',
                                 line=dict(color='cyan', dash='dash'),
                                 text=['', f'{predicted_prices_gru[i]:.2f}'],
                                 textposition='top right'))

    # Vẽ Moving Averages
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['Short MA'],
                             mode='lines', name='Short MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['Long MA'],
                             mode='lines', name='Long MA', line=dict(color='pink')))
    
    # Vẽ EMA
    fig.add_trace(go.Scatter(x=df["Open time"].dt.tz_localize(None), y=df['EMA'],
                             mode='lines', name='EMA', line=dict(color='orange')))

    # Cập nhật bố cục của biểu đồ
    fig.update_layout(
        title=f"Biểu đồ giá {symbol}",
        xaxis_title="Thời gian",
        yaxis_title="Giá (USD)",
        template="plotly_dark",
        width=1200,  # Đặt chiều rộng của biểu đồ
        height=600   # Đặt chiều cao của biểu đồ
    )

    # Hiển thị biểu đồ
    st.plotly_chart(fig)
        # Hiển thị biểu đồ
   
