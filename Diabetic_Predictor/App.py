from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')  # Render the input form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = [
            float(request.form['feature1']),  # Pregnancies
            float(request.form['feature2']),  # Glucose
            float(request.form['feature3']),  # Blood Pressure
            float(request.form['feature4']),  # Skin Thickness
            float(request.form['feature5']),  # Insulin
            float(request.form['feature6']),  # BMI
            float(request.form['feature7']),  # Diabetes Pedigree Function
            float(request.form['feature8'])   # Age
        ]
        input_array = np.array([features])

        # Scale the input features
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Redirect to result page with prediction
        result = "You have diabetes. Please consult a doctor." if prediction == 1 else "You do not have diabetes."
        return render_template('result.html', result_text=result)
    except Exception as e:
        return render_template('result.html', result_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
