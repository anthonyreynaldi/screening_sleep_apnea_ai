from collections import Counter
import json
from flask import Flask, request, jsonify
from preprocessing_step import *
from training_step import *
from prediction_step import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_data(data: dict):
  input_data = {
    'Jenis Kelamin' : data.get('jenis_kelamin', 'Laki-laki'), 
    'Usia (thn)': data.get('usia', 0), 
    'Tinggi (cm)': data.get('tinggi_badan', 0), 
    'Berat (Kg)': data.get('berat_badan', 0),
    'BMI (Kg/m2)': data.get('berat_badan', 0) / (data.get('tinggi_badan', 1)) ** 2,
    'Lingkar Perut (cm)': data.get('lingkar_perut', 0),
    'Lingkar Leher (cm)': data.get('lingkar_leher', 0),
    'Suara Mengorok': data.get('suara_mengorok', 'Pelan'),
    'Terbangun (berapa kali): buang air kecil': data.get('terbangun_buang_air_kecil', 0),
    'Terbangun (berapa kali): tersedak': data.get('terbangun_tersedak', 0),
    'Durasi tidur (jam)': data.get('durasi_tidur', 0),
    'Ngantuk saat beraktifitas': data.get('mengantuk_saat_beraktifitas', 0),
    'Kondisi yang menyertai: Diabetes': "Diabetes" in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Hipertensi': "Hipertensi" in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Hyperkolesterol': 'Hyperkolesterol'  in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Adenoid': 'Adenoid' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Amandel': 'Amandel' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Autoimun': 'Autoimun' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Batuk': 'Batuk' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Bibir Sumbing': 'Bibir Sumbing' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Gerd': 'Gerd' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Hamil': 'Hamil' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Hypertiroid': 'Hypertiroid' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Pilek': 'Pilek' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Polip': 'Polip' in data.get('kondisi_menyertai', []),
    'Kondisi yang menyertai: Lainnya_Rhintis Alergi': 'Rhintis Alergi' in data.get('kondisi_menyertai', []),
  }

  df = pd.DataFrame([input_data])

  numerical_column = get_numerical_column()

  df_prep = numerical_prep(df, numerical_column)
  df_prep = categorical_prep(df_prep)

  return df_prep.iloc[0]


def predict(data: dict):

  #preprocess the data first
  prep_data = preprocess_data(data)

  print(prep_data)

  #find prediction for every model
  models = ['KNN', 'LogisticRegression', 'RandomForest', 'SVM']
  predict_result = {}

  for model in models:
    predict_result[model] = model_predict(model, prep_data)[0]

  #find the most answer predicted from every model
  value_counts = Counter(predict_result.values())
  most_common_value, most_common_count = value_counts.most_common(1)[0]

  prediction = {
    "name": most_common_value.lower(),
    "label_detail": predict_result
  }

  return prediction

@app.route("/predict", methods=["POST"])
def handle_predict():

  if request.method == "POST":

    # Get the data from the request body
    data = request.get_json()
    
    # Check if data is present
    if data is None:
      return jsonify({"error": "Missing data in request body"}), 400
    
    # Perform prediction using the data
    prediction = predict(data)
    
    # Return the prediction as JSON
    return jsonify(prediction)
  
  else:
    return jsonify({"error": "Method not allowed"}), 405
  
@app.route("/retrain", methods=["GET"])
def handle_retrain():
  retrain_models()

  return jsonify({"success": "Retrain Model Success"}), 200

@app.route('/')
def hello_world():
    return 'Hello from Flask Deploy!'

if __name__ == "__main__":
  app.run(debug=True)
