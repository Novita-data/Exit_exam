from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved artifacts
artifacts = joblib.load('model_artifacts.joblib')
model = artifacts['model']
scaler = artifacts['scaler']
feature_columns = artifacts['feature_columns']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        # 1. Get input values from HTML form (Replace with actual feature names)
        input_data = {
            'feature1': float(request.form['feature1']),
            'feature2': float(request.form['feature2']),
            'category_col': request.form['category_col']
        }

        # 2. Convert to DataFrame & Apply One-Hot Encoding + Column Alignment
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # 3. Scale & Predict
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        prediction_text = f"Predicted Result: {pred}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)