import joblib
import warnings
from flask import Flask
from flask import request
from flask import jsonify


model = joblib.load('classifier.pkl')
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
  
    data_values = [
        data['Age'],
        data["C"],
        data["Fare"],
        data["Parch"],
        data["Pclass"],
        data["Q"],
        data["S"],
        data["SibSp"],
        data["female"],
        data["male"],
    ]

    prediction = model.predict([data_values])[0]
    prediction = int(prediction)
    if prediction == 1:
        msg = 'El pasajero sobreviviria' 
    else:
        msg = 'el pasajero no sobreviviria'

    return jsonify({
        'msg': msg,
        'prediction': prediction
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)
