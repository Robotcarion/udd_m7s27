import joblib
import warnings


model = joblib.load('classifier.pkl')
warnings.filterwarnings('ignore')

def main():

    data = {
        "Age": 27.0,
        "C":  0,
        "Fare": 71.2833,
        "Parch": 2,
        "Pclass": 3,
        "Q":  0,
        "S": 1,
        "SibSp": 3,
        "female":  1,
        "male":  0
    }
    
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
    if prediction == 1:
        print('El pasajero sobreviviria') 
    else:
        print('el pasajero no sobreviviria')



if __name__ == "__main__":
    main()
