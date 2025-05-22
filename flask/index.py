from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model("lstm_model.keras")

@app.route('/predict-bulk', methods=['POST'])
def predict_bulk_lstm():
    data = request.get_json()
    current_stock = data['current_stock']
    history = pd.DataFrame(data['history'])
    history['date'] = pd.to_datetime(history['date'])
    history['week'] = history['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly = history.groupby('week').sum(numeric_only=True) # asumsi kolom produk sudah terpisah dan jumlah stock_used per produk
    weekly = weekly.tail(4)

    if len(weekly) < 4:
        return jsonify({"error": "Data harus memiliki minimal 4 minggu riwayat penggunaan"}), 400

    # Normalisasi per kolom produk
    scalers = {}
    scaled_weekly = weekly.copy()
    for col in weekly.columns:
        scaler = MinMaxScaler()
        scaled_weekly[col] = scaler.fit_transform(weekly[[col]])
        scalers[col] = scaler

    # Buat input shape (1, 4, num_produk)
    sequence = np.expand_dims(scaled_weekly.values, axis=0)

    pred_scaled = model.predict(sequence)[0]  # output shape (num_produk, )

    result = {}
    for i, product in enumerate(weekly.columns):
        forecast = scalers[product].inverse_transform([[pred_scaled[i]]])[0][0]
        safety_stock = 0.2 * forecast
        reorder_level = forecast + safety_stock
        current = float(current_stock.get(product, 0))
        stock_to_order = max(0, round(reorder_level - current))

        result[product] = {
            "forecasted_usage": round(forecast),
            "current_stock": round(current),
            "safety_stock": round(safety_stock),
            "reorder_level": round(reorder_level),
            "stock_to_order": stock_to_order
        }

    return jsonify(result)


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
        "product_name": name,
        "average_daily_usage": round(avg_daily_use, 2),
        "current_stock": round(current_stock),
        "predicted_days_until_stockout": round(predicted_days_left, 2),
        "recommended_reorder_date": reorder_date.strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
