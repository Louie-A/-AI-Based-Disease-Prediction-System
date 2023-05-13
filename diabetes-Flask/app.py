from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

classifier = pickle.load(open('../diabetes-Flask/diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("diabetes.html")

@app.route('/display', methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        Preganncies = int(request.form["pregnancies"])
        Glucose = int(request.form["glucose"])
        BloodPressure = int(request.form["bloodpressure"])
        SkinThickness = int(request.form["skinthickness"])
        Insulin = int(request.form["insulin"])
        BMI = int(request.form['bmi'])
        Age = int(request.form['age'])
        DiabetesPredigreeFunction = float(request.form['diabetespredigreefunction'])

        input_data = (Preganncies,Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPredigreeFunction, Age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = classifier.predict(input_data_reshaped)
        print(prediction)
        if(prediction[0] == 0):
            result = '"You are not diabetic"'
        else:
            result = '"You are diabetic"'

        return render_template("diabetes.html", result = result)

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 8001)