import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


diabetes_data = pd.read_csv('diabetes.csv')
diabetes_pipe = pickle.load(open("Diabetes.pkl", 'rb'))


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetes/predict', methods=['POST'])
def diabetes_predict():
    try:
        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        SkinThickness = request.form.get('SkinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')

        input_data = (
            int(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), int(Age))
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = diabetes_pipe.predict(input_data_reshaped)
        if prediction[0] == 0:
            return render_template('diabetes.html', error="The Person is Not Diabetic")
        else:
            return render_template('diabetes.html', error="The Person is Diabetic")


    except:
        return render_template('diabetes.html', error="Enter appropriate values")


heart_data = pd.read_csv('heart.csv')
heart_pipe = pickle.load(open("Heart.pkl", 'rb'))
sc = StandardScaler()
X = heart_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
sc.fit(X)


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/heart/predict', methods=['POST'])
def heart_predict():
    try:
        age = request.form.get('age')
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')

        input_data = (
            int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang),
            float(oldpeak), int(slope), int(ca), int(thal))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = sc.transform(input_data_reshaped)
        prediction = heart_pipe.predict(std_data)

        if prediction[0] == 0:
            return render_template('heart.html', error="Less Likely to have Heart Disease")
        else:
            return render_template('heart.html', error="High chances of Heart Disease")


    except:
        return render_template('heart.html', error="Enter appropriate values")

@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/liver/predict', methods=['POST'])
def liver_predict():
    try:
        age = request.form.get('age')
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')

        input_data = (
            int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang),
            float(oldpeak), int(slope), int(ca), int(thal))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = sc.transform(input_data_reshaped)
        prediction = heart_pipe.predict(std_data)

        if prediction[0] == 0:
            return render_template('heart.html', error="Less Likely to have Heart Disease")
        else:
            return render_template('heart.html', error="High chances of Heart Disease")


    except:
        return render_template('heart.html', error="Enter appropriate values")



if __name__ == "__main__":
    app.run(debug=True, port=5001)
