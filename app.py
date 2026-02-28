from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.save", "rb"))
scaler = pickle.load(open("transform.save", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/manual")
def manual():
    return render_template("manual_predict.html")

@app.route("/sensor")
def sensor():
    return render_template("sensor_predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        ambient = float(request.form["ambient"])
        coolant = float(request.form["coolant"])
        u_d = float(request.form["u_d"])
        u_q = float(request.form["u_q"])
        motor_speed = float(request.form["motor_speed"])
        i_d = float(request.form["i_d"])
        i_q = float(request.form["i_q"])
        stator_yoke = float(request.form["stator_yoke"])
        stator_winding = float(request.form["stator_winding"])

        features = [[
            ambient, coolant, u_d, u_q,
            motor_speed, i_d, i_q,
            stator_yoke, stator_winding
        ]]

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        return render_template(
            "manual_predict.html",
            prediction_text=f"Predicted Motor Temperature: {prediction:.2f}"
        )

    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    app.run(debug=True)
