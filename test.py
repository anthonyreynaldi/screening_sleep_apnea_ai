from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

def predict(data):
  prediction = {
      "name": "berat",
      "label_detail": {
        "SVM": "BERAT",
        "KNN": "SEDANG",
        "Linear Regression": "Ringan",
      }
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

if __name__ == "__main__":
  app.run(debug=True)
