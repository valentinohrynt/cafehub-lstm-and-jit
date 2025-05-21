from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model("lstm_model.keras")

@app.route('/predict-single', methods=['POST'])
def predict_single():
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

if __name__ == '__main__':
    app.run(debug=True)
