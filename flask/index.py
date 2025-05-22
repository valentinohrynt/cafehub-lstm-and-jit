from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model("lstm_model.keras")

@app.route('/predict-bulk-single', methods=['POST'])
def predict_single_lstm():
    data = request.get_json()
    name = data['name']
    current_stock = float(data['current_stock'])
    history = pd.DataFrame(data['history'])  # date + stock_used

    history['date'] = pd.to_datetime(history['date'])
    history['week'] = history['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly = history.groupby('week')['stock_used'].sum().reset_index()

    # Butuh minimal 4 minggu
    if len(weekly) < 4:
        return jsonify({'error': 'Data harus memiliki minimal 4 minggu riwayat penggunaan'}), 400

    weekly = weekly.set_index('week').tail(4)

    # Normalisasi
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(weekly)
    sequence = np.expand_dims(scaled, axis=0)  # (1, 4, 1)

    pred_scaled = model.predict(sequence)[0][0]
    forecast = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Hitung safety stock dll
    safety_stock = 0.2 * forecast
    reorder_level = forecast + safety_stock
    stock_to_order = max(0, round(reorder_level - current_stock))

    return jsonify({
        "Product": name,
        "Forecasted Usage (Next 7 Days)": round(forecast),
        "current_stock": round(current_stock),
        "Safety Stock": round(safety_stock),
        "Reorder Level": round(reorder_level),
        "Stock to Order": stock_to_order
    })

@app.route('/predict-daily-single', methods=['POST'])
def predict_single_jit():
    data = request.get_json()
    name = data['name']
    current_stock = float(data['current_stock'])
    history = pd.DataFrame(data['history'])  # format: [{date, stock_used}]

    history['date'] = pd.to_datetime(history['date'])
    history = history.sort_values('date')

    if len(history) < 2:
        return jsonify({'error': 'Data harus memiliki minimal 2 tanggal untuk estimasi JIT'}), 400

    # Hitung total pemakaian dan durasi hari
    total_used = history['stock_used'].sum()
    date_range = (history['date'].max() - history['date'].min()).days
    avg_daily_use = total_used / date_range if date_range > 0 else total_used

    if avg_daily_use == 0:
        return jsonify({'error': 'Rata-rata pemakaian harian tidak boleh nol'}), 400

    # Prediksi kapan stok habis
    predicted_days_left = current_stock / avg_daily_use
    reorder_date = history['date'].max() + pd.Timedelta(days=predicted_days_left - 5)  # 5 hari buffer

    return jsonify({
        "Product": name,
        "Average Daily Usage": round(avg_daily_use, 2),
        "Current Stock": round(current_stock),
        "Predicted Days Until Stockout": round(predicted_days_left, 2),
        "Recommended Reorder Date": reorder_date.strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
