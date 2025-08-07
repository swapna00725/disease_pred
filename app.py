from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object   # ✅ Added import

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # ✅ Fixed fever mapping
        data = CustomData(
            fever=int(request.form.get('fever')),
            headache=int(request.form.get('headache')),
            nausea=int(request.form.get('nausea')),
            vomiting=int(request.form.get('vomiting')),
            fatigue=int(request.form.get('fatigue')),
            joint_pain=int(request.form.get('joint_pain')),
            skin_rash=int(request.form.get('skin_rash')),
            cough=int(request.form.get('cough')),
            weight_loss=int(request.form.get('weight_loss')),
            yellow_eyes=int(request.form.get('yellow_eyes')),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        pred_index = predict_pipeline.predict(pred_df)[0]  # Predicted encoded label

        encoding_path = os.path.join('artifacts', 'label_encoder.pkl')
        encod = load_object(file_path=encoding_path)  # Load label encoder

        results = encod.classes_[pred_index]  # Decode to disease name

        print("After Prediction")
        print(f"the prediction is {results}")
        return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)