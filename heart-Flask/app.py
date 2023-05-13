from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("heart.html")

@app.route('/display', methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecq"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = int(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        input_data = (age, sex,	cp,	trestbps, chol,	fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        print(prediction)
        if(prediction[0] == 0):
            result = '"You don`t have heart disease"'
        else:
            result = '"You have heart disease"'

        return render_template("heart.html", result=result)

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 8001)