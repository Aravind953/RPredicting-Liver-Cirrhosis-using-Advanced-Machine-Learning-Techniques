from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = pickle.load(open("rf_acc_68.pkl", "rb"))
    scaler = pickle.load(open("normalizer.pkl", "rb"))
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        input_data = [float(request.form[f'feature{i}']) for i in range(1, 11)]
        print("📥 Received input:", input_data)

        # Preprocess inputs
        scaled_data = scaler.transform([input_data])
        print("📊 Scaled input:", scaled_data)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        result = "Cirrhosis Detected ✅" if prediction == 1 else "No Cirrhosis ❎"
        print("🔍 Prediction result:", result)

    except Exception as e:
        print("❌ Prediction error:", e)
        result = f"Error during prediction: {e}"

    return render_template("inner-page.html", result=result)

# Portfolio route
@app.route('/portfolio')
def portfolio():
    return render_template("portfolio-details.html")

# Start server
if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(debug=True,  use_reloader=False)