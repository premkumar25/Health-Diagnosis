from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
cancer_model = pickle.load(open('cancer_model.pkl', 'rb'))
liver_model = pickle.load(open('liver_model.pkl', 'rb'))
kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


def prediction_fun(features, size):

    to_predict_list = list(features.values())
    to_predict_list = list(map(float, to_predict_list))
    final_features = np.array(to_predict_list).reshape((1, size))
    if size == 5:
        prediction = cancer_model.predict(final_features)
    elif size == 10:
        prediction = heart_model.predict(final_features)
    elif size == 7:
        prediction = liver_model.predict(final_features)
    elif (size == 8) and ('bmi' not in features.keys()):
        prediction = kidney_model.predict(final_features)
    elif (size==8) and ('bmi' in features.keys()):
        prediction = diabetes_model.predict(final_features)
    output = prediction[0]
    return output


@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    output = prediction_fun(features, len(features))
    if output:
        prediction_value = 'Consult a doctor, You have high chances of getting disease'
    else:
        prediction_value = "No need to worry, You are healthy!!"
    return render_template('result.html', prediction_text=prediction_value)


@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/kidney')
def kidney():
    return render_template('kidney.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heart')
def heart():
    return render_template('heart.html')


if __name__ == '__main__':
    app.run(debug=True)

