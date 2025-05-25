from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Yang ini nggak kepakai - START
model = load_model("lstm_model.keras")

@app.route('/predict-bulk', methods=['POST'])
def predict_bulk_lstm():
    data = request.get_json()
    current_stock = data['current_stock']
    history = pd.DataFrame(data['history'])
    history['date'] = pd.to_datetime(history['date'])
    history['week'] = history['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly = history.groupby('week').sum(numeric_only=True) 
    weekly = weekly.tail(4)

    if len(weekly) < 4:
        return jsonify({"error": "Data harus memiliki minimal 4 minggu riwayat penggunaan"}), 400

    scalers = {}
    scaled_weekly = weekly.copy()
    for col in weekly.columns:
        scaler = MinMaxScaler()
        scaled_weekly[col] = scaler.fit_transform(weekly[[col]])
        scalers[col] = scaler

    sequence = np.expand_dims(scaled_weekly.values, axis=0)

    pred_scaled = model.predict(sequence)[0] 

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
# Yang ini nggak kepakai - END

@app.route('/jit-signal-event', methods=['POST'])
def jit_signal_event_handler():
    """
    Endpoint ini menerima status stok saat ini dan parameter JIT dari sistem lain (misalnya Laravel),
    lalu menentukan apakah sinyal replenishment perlu diaktifkan.
    """
    data = request.get_json()

    # --- 1. Validasi Input ---
    # Memeriksa apakah semua data yang diperlukan dikirim oleh Laravel.
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required_fields = ['product_name', 'current_stock', 'signal_point', 'replenish_quantity']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    # --- 2. Ambil & Konversi Data ---
    # Ambil semua data dari request. Tidak ada lagi KANBAN_CONFIG.
    product_name = data['product_name']
    try:
        current_stock = float(data['current_stock'])
        signal_point = float(data['signal_point'])
        replenish_quantity = float(data['replenish_quantity'])
    except (ValueError, TypeError):
        return jsonify({"error": "current_stock, signal_point, and replenish_quantity must be valid numbers."}), 400

    # --- 3. Logika Inti JIT ---
    # Logika perbandingan utamanya tetap sama, tapi sekarang lebih sederhana.
    action_required = "NONE"
    message = f"JIT Check for '{product_name}'. Current Stock: {current_stock}, Signal Point: {signal_point}."

    if current_stock <= signal_point:
        action_required = "INITIATE_JIT_REPLENISHMENT"
        message = (
            f"JIT SIGNAL TRIGGERED for '{product_name}'! "
            f"Current Stock ({current_stock}) is at or below Signal Point ({signal_point}). "
            f"Initiate replenishment of {replenish_quantity} units."
        )

    # --- 4. Siapkan & Kirim Respons ---
    # Buat JSON respons untuk dikirim kembali ke Laravel.
    response_data = {
        "product_name": product_name,
        "current_stock_level": current_stock,
        "jit_signal_point_checked": signal_point,
        "action_required": action_required,
        "jit_replenishment_quantity_recommended": replenish_quantity if action_required == "INITIATE_JIT_REPLENISHMENT" else 0,
        "message": message
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)